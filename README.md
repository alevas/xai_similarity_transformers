# xai_similarity_transformers
The implementation and examples for the paper (Explaining Text Similarity in Transformer Models)[https://arxiv.org/abs/2405.06604], accepted to NAACL 2024.

## Usage instructions
**Python version: 3.11**

### Virtual Environment
Simply create a virtual environment in Python 3.11 and install the packages in the `requirements.txt` file and run a jupyter server:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter notebook
```

## Models tested with the code
As also described in the paper, the following models can be used with the code as it is currently:

* `BERT`
* `mBERT` (multilingual BERT)
* `SBERT` 
* `GPT-Neo` 

## LRP 
The lines relevant to these code modifications are marked with `# xai_impl` in both `src/models/xai_bert.py` and `src/models/xai_gpt_neo.py` modules. The implemented changes are done in the Attention heads, the LayerNorm layers and the GELU activation function (applicable for models based on BERT only).

## Citation

@inproceedings{vasileiou-eberle-2024-explaining,
    title = "Explaining Text Similarity in Transformer Models",
    author = "Vasileiou, Alexandros  and
      Eberle, Oliver",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.435/",
    doi = "10.18653/v1/2024.naacl-long.435",
    pages = "7859--7873",
}
