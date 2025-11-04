# sensory-cokge

This repository supports the published paper "Sensory-CoKGE: A Contextualized Knowledge Graph Embedding Framework Using Language Models for Converting Text-Based Food Attributes into Numerical Representation", now published in *Expert Systems with Applications*.  

The Sensory-CoKGE framework integrates knowledge graphs with language models to transform text-based food attributes into meaningful numerical representations, enabling more precise food similarity analysis and recommendations.  

## Citation

If you use this code in your research, please cite our paper.

```bibtex
@article{chang2026sensory-cokge,
  title = {Sensory-CoKGE: A contextualized knowledge graph embedding framework using language models for converting text-based food attributes into numerical representation},
  journal = {Expert Systems with Applications},
  volume = {299},
  number = {Part C},
  year = {2026},
  issn = {0957-4174},
  doi = {10.1016/j.eswa.2025.130191},
  url = {https://doi.org/10.1016/j.eswa.2025.130191},
  author = {Yu-Tang Chang and Shih-Fang Chen},
}
```

## Repository Contents

The source code implements the Sensory-CoKGE framework, including:
- Core modules for language model fine-tuning and sensory-DAG
- Descriptor embedding generation
- Framework evaluation tools

## Get start

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

### Large models (6-8B, only eval. no fine-tuning)

- LLAMA3
- Gemma2
- Qwen2


