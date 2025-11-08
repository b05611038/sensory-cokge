# Sensory-CoKGE: Knowledge Graph Embedding for Food Attributes

This repository supports the published paper:

**"Sensory-CoKGE: A Contextualized Knowledge Graph Embedding Framework Using Language Models for Converting Text-Based Food Attributes into Numerical Representation"**

Published in: *Expert Systems with Applications*, Volume 299, Part C, 2026

[Read the paper](https://doi.org/10.1016/j.eswa.2025.130191)

## Overview

Sensory-CoKGE integrates knowledge graphs with language models to transform text-based food attributes into meaningful numerical representations. This enables more precise food similarity analysis and recommendations.

## Key Features

- Converts descriptive sensory text into numerical embeddings
- Supports multiple language models (BERT, LLAMA3, Gemma2, Qwen2)
- Includes pretrained models for quick start
- Fine-tuning support for custom food attributes
- Comprehensive evaluation tools

## Prerequisites & Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~5GB disk space (for downloaded models)
- GPU recommended (CUDA-compatible) for faster processing
- CPU-only is supported but slower

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/b05611038/sensory-cokge.git
   cd sensory-cokge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Get started with pretrained models in 5 minutes:

```bash
# Step 1: Setup and load pretrained models
python3 embeddings_from_pretrained.py

# Step 2: Evaluate pretrained models
python3 models_evaluation.py pretrained
```

This generates embeddings from pretrained language models and saves results to `./outputs/pretrained`.

**What just happened:**
- Loaded the descriptor graph for food attributes
- Initialized pretrained language models from Hugging Face
- Generated embeddings ready for analysis
- Computed evaluation metrics

**Next steps:**
- See `demo_sensory_cokge_basic_workflow.ipynb` for interactive examples
- Read the detailed workflow guide below

## Detailed Workflow

There are two main workflows:

### Workflow A: Using Pretrained Models (Baseline)

Use this if you want to evaluate existing models without modification.

#### Step 1: Setup and Load Pretrained Models

```bash
python3 embeddings_from_pretrained.py
```

- **Purpose:** Initialize descriptor graph and load pretrained language models
- **Input:** Descriptor graph configuration
- **Output:** Saved models and embeddings in `./outputs/pretrained/`
- **Runtime:** ~5-10 minutes
- **When to use:** First run, or to reset to baseline

#### Step 2: Evaluate Pretrained Models

```bash
python3 models_evaluation.py pretrained
```

- **Purpose:** Calculate performance metrics for pretrained models
- **Input:** Pretrained embeddings from Step 1
- **Output:** Evaluation metrics (accuracy, F1-score, Silhouette score, etc.)
- **Runtime:** ~5-10 minutes
- **Result:** Baseline performance metrics for all supported models

---

### Workflow B: Fine-tuning Custom Models

Use this if you want to train models on your own data.

#### Step 1: Setup (Same as Workflow A)

```bash
python3 embeddings_from_pretrained.py
```

#### Step 2: Generate Fine-tuning Data

```bash
python3 generate_finetuned_data.py
```

- **Purpose:** Create synthetic training data for fine-tuning
- **Input:** Descriptor graph configuration
- **Output:** Training data in `./data/finetuned_data/`
- **Runtime:** ~2-5 minutes
- **When to use:** Before fine-tuning any models

#### Step 3: Fine-tune Models

Choose one or more models to fine-tune:

```bash
python3 finetune_BERT_by_sequence_classification.py
# OR
python3 finetune_LLAMA3_by_sequence_classification.py
# OR
python3 finetune_Gemma2_by_sequence_classification.py
# OR
python3 finetune_Qwen2_by_sequence_classification.py
```

- **Purpose:** Train language models on sensory data
- **Input:** Training data from Step 2
- **Output:** Fine-tuned models in `./outputs/finetuned/`
- **Runtime:** 30 minutes to 2 hours (depends on model size and GPU)
- **When to use:** When you want models specialized for your data

#### Step 4: Generate Embeddings (Fine-tuned)

```bash
python3 embeddings_from_finetuned.py
```

- **Purpose:** Create embeddings from fine-tuned models
- **Input:** Fine-tuned models from Step 3
- **Output:** Embeddings in `./outputs/finetuned/`
- **Runtime:** ~5-10 minutes

#### Step 5: Evaluate Fine-tuned Models

```bash
python3 models_evaluation.py finetuned
```

- **Purpose:** Evaluate fine-tuned model performance
- **Input:** Fine-tuned embeddings from Step 4
- **Output:** Comparison metrics (Pretrained vs Fine-tuned)
- **Runtime:** ~5-10 minutes
- **Result:** See performance improvement from fine-tuning

## Supported Models

The following language models are currently supported:

| Model | Size | Pretrained | Fine-tuning | Speed | VRAM |
|-------|------|-----------|-------------|-------|------|
| BERT | Small | ✓ | ✓ | Fast | Low |
| LLAMA3 | Large | ✓ | ✓ | Medium | High |
| Gemma2 | Medium | ✓ | ✓ | Fast | Medium |
| Qwen2 | Large | ✓ | ✓ | Medium | High |

**Recommendations:**
- Start with BERT for quick testing
- Use Gemma2 for best speed/performance balance
- Use LLAMA3 or Qwen2 if GPU memory allows (best accuracy)

## Troubleshooting

### "Module not found" Error

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** Run `pip install -r requirements.txt`

Check Python version: `python --version` (must be 3.8+)

### Out of Memory (OOM) Error

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use a smaller model (BERT instead of LLAMA3)
- Use CPU only (see "CPU-only mode" below)
- Close other GPU applications

Check GPU: `nvidia-smi`

### Script Hangs or Takes Too Long

**For `embeddings_from_pretrained.py`:**
- First run downloads models from Hugging Face (~5-10 minutes depending on internet)
- Verify internet connection
- Check disk space: `df -h` (need ~5GB)

**For fine-tuning scripts:**
- Fine-tuning is slow on CPU (use GPU if available)
- LLAMA3/Qwen2 are slower than BERT (expected)

### CPU-Only Mode

To run on CPU without GPU (slower but works):
1. Modify fine-tuning scripts to use CPU device
2. Set batch size smaller to fit in RAM
3. Expected runtime: 5-10x slower than GPU

## FAQ

**Q: Can I use this tool for other foods besides coffee?**

A: Yes! Modify the descriptor graph in `generate_finetuned_data.py` to use your food type and descriptors. The framework is generalizable.

**Q: How long does fine-tuning take?**

A: 30 minutes to 2 hours depending on model size and GPU. BERT is fastest (~30 min), LLAMA3 is slowest (~2 hours).

**Q: Can I use only CPU?**

A: Yes, but it's much slower (5-10x). GPU is highly recommended.

**Q: Where are my results saved?**

A: Embeddings and models are saved in `./outputs/` directory. Results from evaluation are also printed to console.

## Citation

If you use Sensory-CoKGE in your research, please cite:

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

Or plain text:

```
Chang, Y.-T., & Chen, S.-F. (2026). Sensory-CoKGE: A contextualized knowledge
graph embedding framework using language models for converting text-based food
attributes into numerical representation. Expert Systems with Applications,
299(C), 130191. https://doi.org/10.1016/j.eswa.2025.130191
```
