import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor, device, nn
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
import configs
from src.utils.helper import SenEmbGen


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling as implemented by sentence-transformers for usage with
    standard Hugging Face models:
    https://huggingface.co/sentence-transformers/stsb-bert-large
    Args:
        model_output:
        attention_mask:

    Returns:

    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output
    input_mask_expanded = (attention_mask.unsqueeze(-1)
                           .expand(token_embeddings.size())
                           .float()
                           .to(configs.device))

    mean_pooled_output = torch.sum(token_embeddings.to(configs.device) *
                                   input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)
    return mean_pooled_output


class SimilarityTransformer(nn.Module):

    def __init__(self,
                 config,
                 embeddings,
                 encoder,
                 tokenizer,
                 sim_func="cosine"):
        super().__init__()

        self.config_ = config

        self.embeddings = embeddings
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.sim_func = sim_func

    def _pool_encoding(self, all_outs, attention_mask) -> Optional[
        torch.Tensor]:
        """
        Perform a pooling operation on the embeddings of the input sequences.
        Args:
            all_outs: all of the model's outputs
            attention_mask: the attention mask

        Returns:
            the final sentence embedding
        """
        if self.config_.pooling.value == SenEmbGen.MEAN_POOLING.value:
            sentence_embeddings = mean_pooling(all_outs, attention_mask)
        elif self.config_.pooling.value == SenEmbGen.CLS_POOLING.value:
            sentence_embeddings = all_outs[:, 0]
        else:
            raise ValueError(f"Unknown value for {self.config_.pooling}!")
        return sentence_embeddings

    @staticmethod
    def _get_extended_attention_mask(
        attention_mask: Tensor,
        input_shape: tuple,
        device: str = configs.device) -> Optional[
        torch.Tensor]:
        """Makes broadcastable attention mask and causal mask so that future
        and marked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: the device to store stuff

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask
        # .to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def encode(self, input_ids_1=None,
               attention_mask_1=None,
               token_type_ids_1=None,
               position_ids_1=None,
               input_ids_2=None,
               attention_mask_2=None,
               token_type_ids_2=None,
               position_ids_2=None,
               inputs_embeds1=None,
               inputs_embeds2=None,
               labels=None,
               past_key_values_length=0):
        """
        The same as the forward method but without the loss function.
        Args:
            input_ids_1:
            attention_mask_1:
            token_type_ids_1:
            position_ids_1:
            input_ids_2:
            attention_mask_2:
            token_type_ids_2:
            position_ids_2:
            inputs_embeds1:
            inputs_embeds2:
            labels:
            past_key_values_length:

        Returns:

        """
        input1 = {'input_ids': input_ids_1, 'token_type_ids': token_type_ids_1,
                  'attention_mask': attention_mask_1,
                  'inputs_embeds': inputs_embeds1}
        input2 = {'input_ids': input_ids_2, 'token_type_ids': token_type_ids_2,
                  'attention_mask': attention_mask_2,
                  'inputs_embeds': inputs_embeds2}

        X = []

        for x in [input1, input2]:

            attention_mask = x['attention_mask']

            input_shape = x['input_ids'].size()

            extended_attention_mask = self._get_extended_attention_mask(
                attention_mask, input_shape, self.config_.device)
            if x['inputs_embeds'] is not None:
                # Apply bypass trick
                model_inputs = {'input_ids': None,
                                'past_key_values': None,
                                'use_cache': False,
                                'position_ids': torch.tensor([[int(i) for i in
                                                               range(x[
                                                                         'input_ids'].shape[
                                                                         -1])]]).to(
                                    self.config_.device),
                                'attention_mask': x['attention_mask'],
                                'token_type_ids': None}

                position_embeds = self.embeddings(model_inputs['position_ids'])
                embedding_output = position_embeds + x['inputs_embeds']
                type = "gpt"
            else:
                embedding_output = self.embeddings(x['input_ids'])
                type = "bertoid"

            if type == "gpt":
                model_inputs['inputs_embeds'] = embedding_output

                last_hidden_state = self.encoder(inputs_embeds=model_inputs[
                    'inputs_embeds']).last_hidden_state
                sentence_embeddings = self._pool_encoding(last_hidden_state,
                                                          attention_mask=attention_mask.to(
                                                             configs.device))

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    attention_mask=extended_attention_mask,

                    output_hidden_states=False
                )

                attn_input = encoder_outputs['last_hidden_state']

                sentence_embeddings = self._pool_encoding(attn_input,
                                                          attention_mask=attention_mask)

            X.append(sentence_embeddings)

        if self.sim_func == 'cosine':
            sims = cosine_similarity(X[0], X[1])
        elif self.sim_func == 'dot':
            sims = torch.sum(X[0] * X[1], 1)
        else:
            raise

        return dict(
            logits=sims,
            hidden_states=tuple(X),
            similarity_scores=sims
        )

    def forward(self, input_ids_1=None,
                attention_mask_1=None,
                token_type_ids_1=None,
                position_ids_1=None,
                input_ids_2=None,
                attention_mask_2=None,
                token_type_ids_2=None,
                position_ids_2=None,
                inputs_embeds1=None,
                inputs_embeds2=None,
                labels=None,
                past_key_values_length=0):
        """
        Standard forward method used for training the similarity model.

        Args:
            input_ids_1:
            attention_mask_1:
            token_type_ids_1:
            position_ids_1:
            input_ids_2:
            attention_mask_2:
            token_type_ids_2:
            position_ids_2:
            inputs_embeds1:
            inputs_embeds2:
            labels:
            past_key_values_length:

        Returns:

        """
        input1 = {'input_ids': input_ids_1, 'token_type_ids': token_type_ids_1,
                  'attention_mask': attention_mask_1,
                  'inputs_embeds': inputs_embeds1}
        input2 = {'input_ids': input_ids_2, 'token_type_ids': token_type_ids_2,
                  'attention_mask': attention_mask_2,
                  'inputs_embeds': inputs_embeds2}

        X = []

        for x in [input1, input2]:

            attention_mask = x['attention_mask']

            input_shape = x['input_ids'].size()

            extended_attention_mask: torch.Tensor = self._get_extended_attention_mask(
                attention_mask, input_shape,
                self.config_.device)
            if x['inputs_embeds'] is not None:
                # Apply bypass trick
                model_inputs = {'input_ids': None,
                                'past_key_values': None,
                                'use_cache': False,
                                'position_ids': torch.tensor([[int(i) for i in
                                                               range(x[
                                                                         'input_ids'].shape[
                                                                         -1])]]).to(
                                    self.config_.device),
                                'attention_mask': x['attention_mask'],
                                'token_type_ids': None}

                position_embeds = self.embeddings(model_inputs['position_ids'])
                embedding_output = position_embeds + x['inputs_embeds']
                type = "gpt"
            else:
                embedding_output = self.embeddings(x['input_ids'])
                type = "bertoid"

            if type == "gpt":
                model_inputs['inputs_embeds'] = embedding_output

                last_hidden_state = self.encoder(inputs_embeds=model_inputs[
                    'inputs_embeds']).last_hidden_state
                sentence_embeddings = self._pool_encoding(last_hidden_state,
                                                          attention_mask=attention_mask.to(
                                                             configs.device))

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    attention_mask=extended_attention_mask,

                    output_hidden_states=False
                )

                attn_input = encoder_outputs['last_hidden_state']

                sentence_embeddings = self._pool_encoding(attn_input,
                                                          attention_mask=attention_mask)

            X.append(sentence_embeddings)
        if self.sim_func == 'cosine':
            sims = cosine_similarity(X[0], X[1])
        elif self.sim_func == 'dot':
            sims = torch.sum(X[0] * X[1], 1)
        else:
            raise

        loss = nn.functional.mse_loss(sims, labels)

        return SequenceClassifierOutput(loss=loss,
                                        logits=sims,
                                        hidden_states=tuple(X)
                                        )

    def forward_explain(self, input_ids_1=None,
                        attention_mask_1=None,
                        token_type_ids_1=None,
                        position_ids_1=None,
                        input_ids_2=None,
                        attention_mask_2=None,
                        token_type_ids_2=None,
                        position_ids_2=None,

                        inputs_embeds1=None,
                        inputs_embeds2=None,
                        labels=None,
                        past_key_values_length=0,
                        sim_func='cosine'):
        """
        Performs a forward pass as well as the BiLRP backwardpass on the data.
        If encoder.lrp==False, then the backward pass computes the HxP
        explanation.

        Args:
            input_ids_1:
            attention_mask_1:
            token_type_ids_1:
            position_ids_1:
            input_ids_2:
            attention_mask_2:
            token_type_ids_2:
            position_ids_2:
            inputs_embeds1:
            inputs_embeds2:
            labels:
            past_key_values_length:
            sim_func:

        Returns:

        """
        input1 = {'input_ids': input_ids_1, 'token_type_ids': token_type_ids_1,
                  'attention_mask': attention_mask_1,
                  'inputs_embeds': inputs_embeds1}
        input2 = {'input_ids': input_ids_2, 'token_type_ids': token_type_ids_2,
                  'attention_mask': attention_mask_2,
                  'inputs_embeds': inputs_embeds2}

        X = []
        Rsens = []

        for x in tqdm([input1, input2]):

            attention_mask = x['attention_mask']
            input_shape = x['input_ids'].size()
            extended_attention_mask: torch.Tensor = self._get_extended_attention_mask(
                attention_mask, input_shape,
                self.config_.device)

            if x['inputs_embeds'] is not None:
                # Apply bypass trick
                model_inputs = {'input_ids': None,
                                'past_key_values': None,
                                'use_cache': False,
                                'position_ids': torch.tensor([[int(i) for i in
                                                               range(x[
                                                                         'input_ids'].shape[
                                                                         -1])]]).to(
                                    self.config_.device),
                                'attention_mask': x['attention_mask'],
                                'token_type_ids': None}

                position_embeds = self.embeddings(model_inputs['position_ids'])
                embedding_output = position_embeds + x['inputs_embeds']
                global_model_grad = True
            else:
                embedding_output = self.embeddings(x['input_ids'])
                global_model_grad = False

            Rsen = []
            for j in range(self.config_.hidden_size):
                logging.debug("PERFORMING FORWARD METHOD")

                if global_model_grad == False:
                    # Compute explanations by iterating through modules
                    encoder_outputs = self.encoder.forward_explain(
                        embedding_output,
                        attention_mask=extended_attention_mask
                    )
                    attn_input = encoder_outputs['last_hidden_state'].to(
                        configs.device)
                    sentence_embeddings = self._pool_encoding(attn_input,
                                                              attention_mask=attention_mask.to(
                                                                 configs.device))

                    logging.debug(f"EXPLAINING")
                    A = encoder_outputs['A']
                    sentence_embeddings_data = sentence_embeddings.data.squeeze()
                    sentence_embeddings_data.requires_grad_(True)

                    sen_feature = sentence_embeddings_data[j]
                    sen_feature.backward()  # this causes grad calculation to sentence_embeddings_data, sentence_embeddings_data.grad is 0 apart from the j-th element

                    R0 = (sentence_embeddings_data.grad * sentence_embeddings)
                    # at this point Rsen_ = R0 for feature j, where all elements are 0 apart from the j-th element which is
                    # the same as sen_feature

                    Rsen_ = R0
                    # iterate through the layers in reverse order
                    for i, _ in list(enumerate(self.encoder.layer))[::-1]:
                        Rsen_.sum().backward()
                        Rsen_attn_ = (
                            (A['attn_input_{}_data'.format(i)].grad) * A[
                            'attn_input_{}'.format(i)])

                        Rsen_ = Rsen_attn_

                else:
                    # Computing global model gradients.
                    # Since we want to compute the gradient wrt to the Transformer, 
                    # be careful with position and token embeddings, the position embeddings 
                    # are now bypassed in the encoder, since they have already 
                    # been added to the inputs_embeds (see above)

                    embeddings_ = embedding_output.detach().requires_grad_(
                        True)
                    model_inputs['inputs_embeds'] = embeddings_

                    last_hidden_state = self.encoder(
                        inputs_embeds=model_inputs[
                            'inputs_embeds']).last_hidden_state
                    sentence_embeddings = self._pool_encoding(last_hidden_state,
                                                              attention_mask=attention_mask.to(
                                                                 configs.device))

                    # Select what signal to propagate back through the network
                    selected_logit = sentence_embeddings[:, j]
                    selected_logit.backward()

                    gradient = embeddings_.grad
                    relevance = gradient * embeddings_
                    Rsen_ = relevance

                Rsen.append(Rsen_.sum(2).detach().cpu().numpy())

            X.append(sentence_embeddings)
            Rsens.append(np.asarray(Rsen))

        relevance_scores = Rsens[0].squeeze().T @ Rsens[1].squeeze()

        if sim_func == 'cosine':
            sims = cosine_similarity(X[0], X[1])
        elif sim_func == 'dot':
            sims = torch.sum(X[0] * X[1], 1)
        else:
            raise
        relevance_scores /= np.abs(relevance_scores).max()

        return {
            "sentence_embeddings": X,
            "relevance_scores": relevance_scores,
            "similarity_scores": sims,
            "Rsen": Rsens}

    def forward_explain_embeddings(self, input_ids_1=None,
                                   attention_mask_1=None,
                                   token_type_ids_1=None,
                                   position_ids_1=None,
                                   input_ids_2=None,
                                   attention_mask_2=None,
                                   token_type_ids_2=None,
                                   position_ids_2=None,
                                   inputs_embeds=None,
                                   labels=None,
                                   past_key_values_length=0):
        """
        Another explanation method but only uses the output of the embeddings
        for a given input, disregarding the encoder. Part of the benchmarks
        of the paper.
        Args:
            input_ids_1:
            attention_mask_1:
            token_type_ids_1:
            position_ids_1:
            input_ids_2:
            attention_mask_2:
            token_type_ids_2:
            position_ids_2:
            inputs_embeds:
            labels:
            past_key_values_length:

        Returns:
            object:

        """
        input1 = {'input_ids': input_ids_1, 'token_type_ids': token_type_ids_1,
                  'attention_mask': attention_mask_1}
        input2 = {'input_ids': input_ids_2, 'token_type_ids': token_type_ids_2,
                  'attention_mask': attention_mask_2}

        X = []
        Rsens = []
        for x in tqdm([input1, input2]):
            embedding_output = self.embeddings(x['input_ids'])

            attention_mask = x['attention_mask']

            Rsen = []
            sentence_embeddings = self._pool_encoding(
                embedding_output.to(configs.device),
                attention_mask=attention_mask.to(configs.device))
            R0 = embedding_output
            Rsen_ = R0
            Rsen.append(Rsen_.detach().cpu().numpy())

            X.append(sentence_embeddings)
            Rsens.append(np.asarray(Rsen))
        relevance_scores = np.square(Rsens[0].squeeze() @ Rsens[
            1].squeeze().T)  ## MATH squaring to match the units
        relevance_scores /= np.abs(relevance_scores).max()
        sims = X[0] @ X[1].T

        return {
            "sentence_embeddings": X,
            "relevance_scores": relevance_scores,
            "similarity_scores": sims,
            "Rsen": Rsens}

    def reset_biases(self) -> None:
        """
        Sets the biases of the model to zero. Used to verify the conservation
        property of BiLRP.
        """
        state_dict_zero_bias = self.state_dict()

        for k, v in state_dict_zero_bias.items():
            if '.bias' in k:
                state_dict_zero_bias[k] = 0. * v

            elif 'masked_bias' in k:
                state_dict_zero_bias[k] = 0. * v
        self.load_state_dict(state_dict_zero_bias)
