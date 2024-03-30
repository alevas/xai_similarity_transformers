from enum import Enum
from typing import Dict, List

from unidecode import unidecode

import configs


def format_tokenizer_outs_xai(encoded_input1: Dict,
                              encoded_input2: Dict,
                              encoder=None) -> Dict:
    """
    An adapter method for transforming the tokenizer output to the format
    the original BERT and GPT code was written.
    Args:
        encoded_input1: the first sentence tokenizer output
        encoded_input2: the second sentence tokenizer output
        encoder: for the GPT XAI model. We retrieve the encoder.wte output
            in an intermediate inputs_embeds param

    Returns: the 'united' tokenizer data for the Similarity Transformer model

    """
    encoded_input_united = {}
    for key, val in encoded_input1.items():
        encoded_input_united[key + "_1"] = val.to(configs.device)
    for key, val in encoded_input2.items():
        encoded_input_united[key + "_2"] = val.to(configs.device)

    if encoder:  # for GPT
        encoded_input_united['inputs_embeds1'] = encoder.wte(
            encoded_input_united['input_ids_1'])
        encoded_input_united['inputs_embeds2'] = encoder.wte(
            encoded_input_united['input_ids_2'])

    return encoded_input_united


def preprocess_sentence(sentence: List) -> List:
    """
    Preprocess sentences by replacing special characters or double spaces.
    May be needed for some STSb samples.
    Args:
        sentence: the sentence string to be preprocessed

    Returns: the preprocessed sentence

    """
    list_str_to_replace = ["‘", "—"]
    for to_replace in list_str_to_replace:
        sentence = sentence.replace(to_replace, unidecode(to_replace))
    sentence = sentence.replace("  ", " ")
    return sentence


class SenEmbGen(Enum):
    MEAN_POOLING = "mean_pooling"
    CLS_POOLING = "cls_pooling"
