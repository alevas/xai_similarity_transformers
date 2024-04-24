# xai_similarity_transformers
Explaining Text Similarity in Transformer Models 

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
The BiLRP implementation is TODO
The lines relevant to these code modifications are marked with `# xai_impl` in both `src/models/xai_bert.py` and `src/models/xai_gpt_neo.py` modules. The implemented changes are done in the Attention heads, the LayerNorm layers and the GELU activation function (applicable for models based on BERT only).
