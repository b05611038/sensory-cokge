# language_models_embedding
The research code of using LLMs to quantify text info in sensory evaluation

## Get start

Please download private library: [coffee_database] (http://gofile.me/3Rex9/rZ8SoE5iz) first.  
Please see the README.md in the folder and use pip to install it first.
After installation, use following command to install necessary modules.  

```bash
pip3 install -r requirements.txt
```

## Usage

Here is some basic usages.

### Setup Descriptor graph and pretrained models

Please run this command first to setup needed objects.  
In this script, the pretrained model will be loaded.  
Including the descriptor graph, all needed object will save in ./outputs

```python3
python3 embeddings_from_pretrained.py 
```

### Evaluate pretrained LLMs

This command can evalaute the pretrained models released by HuggingFace transformers.  
Note, not all transofoemers model is implemented.  

```python
python3 models_evaluation.py pretrained
```

### Fintuning the LLMs

Before finetuning LLMs, please use following command to generate synthetic data.  

```python
python3 generate_finetuned_data.py
```

After data generation, can use following commands to finetune LLMs.  

```python
python3 finetune_[MODEL]_by_sequence_classification.py 
```

### Evaluate finetuned LLMs

This command can evalaute the finetuned models released by HuggingFace transformers.
The finetuend model please refer Fintuning the LLMs part.  

```python3
python3 embeddings_from_finetuned.py 
python3 models_evaluation.py finetuned 
```

## Needed developed models

### Large models (6-8B, latest)

- LLAMA3
- Gemma2
- Qwen2


