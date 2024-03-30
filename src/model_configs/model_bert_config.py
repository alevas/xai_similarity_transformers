import configs

device = configs.device
from transformers.models.bert.configuration_bert import BertConfig


class BERTXAIConfig(BertConfig):
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.layer_norm_eps = 1e-12
        self.n_classes = 5
        self.num_hidden_layers = 12

        self.attention_head_size = int(
            self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.detach_layernorm = True  # Detaches the attention-block-output LayerNorm
        self.detach_kq = True  # Detaches the kq-softmax branch
        self.detach_mean = False
        self.device = device
        self.train_mode = False

        ########## Custom Bert encoder params ##########
        super().__init__(output_attentions=False,
                         # default BERT encoder value: False
                         output_hidden_states=False,
                         # default BERT encoder value: False
                         attention_probs_dropout_prob=0.1,
                         # default BERT encoder value: 0.1
                         hidden_dropout_prob=0.1,
                         # default BERT encoder value: 0.1
                         intermediate_size=3072,
                         # default BERT encoder value: 3072
                         hidden_act='gelu',
                         # default BERT encoder value: 'gelu'
                         is_decoder=False,  # default BERT encoder value: False
                         is_encoder_decoder=False,
                         # default BERT encoder value: False
                         chunk_size_feed_forward=0,
                         # default BERT encoder value: 0
                         add_cross_attention=False)  # default BERT encoder value: False)
