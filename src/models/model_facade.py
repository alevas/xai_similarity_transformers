import logging
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel

import configs
from src.model_configs.model_bert_config import BERTXAIConfig
from src.models.similarity_transformer import SimilarityTransformer
from src.models.xai_bert import BertEncoderXAI
from src.models.xai_gpt_neo import GPTNeoModelXAI
from src.utils.helper import SenEmbGen


class ModelFacade:
    def __init__(self,
                 model_name,
                 finetuned_model_path=None,
                 train_mode=False,
                 pooling="MEAN_POOLING",
                 reset_biases=False,
                 freeze_encoder=True,
                 freeze_embeddings=True,
                 lrp=True,
                 type="bert",
                 sentence_transformer=False,
                 max_padding=70,
                 override_weights=False,
                 no_of_layers=12,
                 sim_func="cosine"
                 ):
        self.no_of_layers = no_of_layers
        self.model_name = model_name
        self.train_mode = train_mode
        self.reset_biases = reset_biases
        self.model_type = type
        self.logger = logging.getLogger()
        self.pooling = pooling
        self.finetuned_model_path = finetuned_model_path
        self.tokenizer = self._load_tokenizer()
        self.sentence_transformer = sentence_transformer
        self.override_weights = override_weights
        self.sim_func = sim_func
        if sentence_transformer:
            self.base_model = self._load_sentence_transformer()

        else:
            self.base_model = self._load_auto_model()

        if type == "bert":
            config = BERTXAIConfig()
            self.embeddings = self.base_model.embeddings
        elif type == "gpt":
            config = self.base_model.config
            self.embeddings = self.base_model.wpe
        else:
            raise NotImplementedError(
                "Only BERT and GPT models supported currently!")

        self.embeddings.to(configs.device)
        config.train_mode = train_mode
        config.device = configs.device
        config.pooling = SenEmbGen[pooling]

        self._prepare_base_model(config=config,
                                 freeze_embeddings=freeze_embeddings,
                                 freeze_encoder=freeze_encoder,
                                 lrp=lrp)

        self.sim_model = self._load_sim_model(config, max_padding)

    def __call__(self, *args, **kwargs):
        return self.sim_model

    def _prepare_base_model(self, freeze_embeddings, freeze_encoder, config,
                            lrp):
        if not self.train_mode:
            self.logger.debug("Loading model for XAI stuff.")

            if self.model_type == "bert":
                encoder = BertEncoderXAI(config=config, lrp=lrp)
                encoder.load_state_dict(self.base_model.encoder.state_dict())

            elif self.model_type == "gpt":
                encoder = GPTNeoModelXAI(config=config, lrp=lrp)
                encoder.load_state_dict(self.base_model.state_dict())
                encoder.wpe = Zero(config.hidden_size)

            try:  # automodel GPT is too large for my poor GPU
                encoder.to(configs.device)
            except torch.cuda.OutOfMemoryError:
                self.logger.error(
                    "Not enough CUDA memory. Reverting all to CPU!!!")
                configs.device = torch.device('cpu')
            self.encoder = encoder

        else:
            self.logger.warning("EXECUTING IN TRAIN MODE!")
            self.encoder = self.base_model.encoder
        if self.model_type == "bert" and self.no_of_layers != len(
            encoder.layer):  # less layers than default for some reason
            encoder.layer = encoder.layer[:self.no_of_layers]
        if freeze_embeddings:  # todo is this working for GPT?
            for param in self.embeddings.parameters():
                param.requires_grad = False
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_auto_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        model.to(configs.device)

        return model

    def _load_sentence_transformer(self):
        model = SentenceTransformer(self.model_name)
        model.to(configs.device)

        return model[0].auto_model

    def _load_sim_model(self, config, max_padding):
        sim_model = SimilarityTransformer(config=config,
                                          embeddings=self.embeddings,
                                          encoder=self.encoder,
                                          tokenizer=self.tokenizer,
                                          sim_func=self.sim_func)
        sim_model.to(configs.device)
        if self.finetuned_model_path:
            self.logger.info(
                f"Loading finetuned model {self.finetuned_model_path}")
            dict_ = torch.load(
                os.path.join(configs.ROOT_DIR, self.finetuned_model_path))
            if self.override_weights == "sbert_sa":  # ok this is tiring
                dict_["readout.W1.weight"] = dict_["1.W1.weight"]
                dict_["readout.W2.weight"] = dict_["1.W2.weight"]
                dict_new = {}
                del dict_["0.auto_model.pooler.dense.weight"]
                del dict_["0.auto_model.pooler.dense.bias"]
                del dict_["1.W1.weight"]
                del dict_["1.W2.weight"]
                for key, val in dict_.items():
                    dict_new[key.replace("0.auto_model.", "")] = val
                dict_ = dict_new
            try:
                sim_model.load_state_dict(dict_)
            except (RuntimeError, KeyError) as e:
                logger = logging.getLogger()
                logger.error("Could not load the state dict!")
                logger.error(
                    f"Keys in current dict: " + ", ".join((dict_.keys())))
                logger.error(e)
        if not self.train_mode:
            sim_model.eval()
        if self.reset_biases:
            sim_model.reset_biases()
        return sim_model

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)


class Zero(nn.Identity):
    """A layer that just returns Zero-Embeddings"""

    def __init__(self, dim=768, *args: Any, **kwargs: Any) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros((input.shape[0], input.shape[1], self.dim)).to(
            input.device)
