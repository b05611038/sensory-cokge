# Sensory-CoKGE: Knowledge Graph Embedding for Food Sensory Attributes

This repository supports the published paper:

**"Sensory-CoKGE: A Contextualized Knowledge Graph Embedding Framework Using Language Models for Converting Text-Based Food Attributes into Numerical Representation"**

Published in: *Expert Systems with Applications*, Volume 299, Part C, 2026

[Read the paper](https://doi.org/10.1016/j.eswa.2025.130191)

## Overview

Sensory-CoKGE converts descriptive food sensory text (e.g., "nutty", "acidic", "smooth") into numerical vector representations called embeddings. These embeddings enable **statistical analysis of sensory attributes** - a crucial need in food science and sensory analysis where understanding how attributes relate to and vary with each other is more important than generating text.

By using pretrained language models, the framework captures semantic meaning in sensory descriptors, creating embeddings suitable for clustering, correlation analysis, dimensionality reduction, and other statistical methods.

## Key Features

- Converts text-based food sensory attributes into numerical embeddings
- Supports 6 fine-tunable models optimized for computers with limited resources
- Optionally integrates large language models (8B) for advanced analysis
- Produces embeddings ready for statistical analysis and visualization
- Provides evaluation metrics for model comparison
- Works with regular computers (CPU or modest GPU)

## Prerequisites & Installation

### System Requirements

- **Python:** 3.8 or higher
- **Disk Space:** ~2GB for small models, ~10GB if including large models (8B)
- **RAM:** 8GB minimum (16GB+ recommended)
- **GPU:** Optional but improves performance (any modern GPU with 8GB+ VRAM)
- **CPU-only mode:** Fully supported, just slower

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/b05611038/sensory-cokge.git
   cd sensory-cokge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Quick Start (10 minutes)

Start with pretrained small models - no fine-tuning needed:

```bash
# Step 1: Generate embeddings from pretrained models
python3 embeddings_from_pretrained.py

# Step 2: Evaluate model performance
python3 models_evaluation.py pretrained

# Step 3 (Optional): Visualize embeddings
# See Notebook_embedding_visualization.ipynb for interactive examples
```

This will:
- Load the food sensory descriptor knowledge graph
- Generate embeddings using pretrained BERT, RoBERTa, ALBERT, BART, GPT-2, and T5
- Save embeddings in `./outputs/` for analysis
- Print evaluation metrics comparing model performance

**What are embeddings?**
Embeddings are numerical vectors that capture the meaning of sensory descriptors. Similar descriptors (e.g., "smooth" and "creamy") will have similar embeddings, enabling statistical grouping and analysis.

## Detailed Workflow

### Workflow A: Using Pretrained Models (Recommended for Most Users)

Best for food researchers who want immediate analysis without model training.

**Step 1: Generate Pretrained Embeddings**

```bash
python3 embeddings_from_pretrained.py
```

- **What it does:** Loads descriptor graph and generates embeddings using pretrained language models
- **Output:** Embeddings saved in `./outputs/pretrained-[MODEL]_embeddings.pkl`
- **Runtime:** ~5-10 minutes (first run downloads models from internet)
- **Result:** Ready-to-use embeddings for all 6 models

**Step 2: Evaluate Model Performance**

```bash
python3 models_evaluation.py pretrained
```

- **What it does:** Computes metrics showing how well each model preserves sensory attribute relationships
- **Output:** Metrics printed to console and optionally saved to CSV
- **Metrics included:**
  - Adjacency matching (structural preservation)
  - Distance matching (relationship preservation)
  - L2 and angular distance measures
- **Runtime:** ~5-10 minutes

**Step 3: Visualize Embeddings**

```bash
jupyter notebook Notebook_embedding_visualization.ipynb
```

- Interactive visualization of embeddings using t-SNE and PCA
- See how similar descriptors cluster together
- Compare different models side-by-side

---

### Workflow B: Fine-tuning Models on Your Own Data

For researchers who want to specialize models on their own sensory descriptors.

**When to use fine-tuning:**
- You have domain-specific descriptors not well represented in general text
- You want models optimized for your specific food category
- You need better embeddings for your statistical analysis

**Prerequisites:**
- Complete Workflow A Step 1 first
- Moderate computational resources (2-4 hours on GPU, or overnight on CPU)

#### Step 1: Generate Fine-tuning Data

```bash
python3 generate_finetuned_data.py
```

- **What it does:** Creates synthetic training pairs from the descriptor graph
- **Output:** Training files in `./outputs/`
- **Runtime:** ~2-5 minutes
- **Default:** 100,000 training samples, 10,000 evaluation samples
- **Customization:** Use `--train_sample_number` and `--eval_sample_number` to adjust

#### Step 2: Fine-tune a Model

Choose one model to fine-tune:

```bash
# Small model - fastest (30 min on GPU, 4-6 hours on CPU)
python3 finetune_BERT_by_sequence_classification.py

# Alternative models:
python3 finetune_RoBERTa_by_sequence_classification.py
python3 finetune_ALBERT_by_sequence_classification.py
python3 finetune_GPT2_by_sequence_classification.py
python3 finetune_BART_by_sequence_classification.py
python3 finetune_T5_by_sequence_classification.py
```

- **What it does:** Trains model on sensory descriptor pairs to improve embeddings
- **Input:** Generated training data from Step 1
- **Output:** Fine-tuned model saved to `./outputs/`
- **Runtime:** 30 min - 4 hours depending on model and hardware
- **Result:** Model specialized for your descriptors

#### Step 3: Generate Fine-tuned Embeddings

```bash
python3 embeddings_from_finetuned.py
```

- **What it does:** Creates embeddings using your fine-tuned model
- **Output:** Embeddings in `./outputs/`
- **Runtime:** ~5-10 minutes

#### Step 4: Compare Fine-tuned vs Pretrained

```bash
python3 models_evaluation.py finetuned
```

- **What it does:** Evaluates your fine-tuned model and compares with pretrained baseline
- **Output:** Metrics showing improvement from fine-tuning
- **Runtime:** ~5-10 minutes

---

## Available Models

### Small Models (Fine-tunable) - Recommended for Most Users

These models can be both evaluated and fine-tuned. They work well on regular computers.

| Model | Model Size | Speed | VRAM | Fine-tuning Time (GPU) | Best For |
|-------|-----------|-------|------|----------------------|----------|
| BERT | 110M | Very Fast | 2GB | 20-30 min | Quick baseline |
| RoBERTa | 355M | Fast | 4GB | 30-40 min | Better accuracy |
| ALBERT | 12M | Very Fast | 1GB | 15-20 min | Limited resources |
| GPT-2 | 124M | Fast | 2GB | 30-40 min | Alternative approach |
| BART | 406M | Fast | 4GB | 40-60 min | Sequence-to-sequence |
| T5 | 220M | Fast | 4GB | 40-60 min | Flexible architecture |

**Recommendation:** Start with **BERT** for quick testing, then try **RoBERTa** for better results.

### Large Models (8B) - Evaluation Only

These models provide state-of-the-art embeddings but require more resources and **cannot be fine-tuned** with this code.

```bash
# Optional: Include large models in pretrained evaluation
python3 embeddings_from_pretrained.py --enable_LLM
```

| Model | Model Size | Speed | VRAM | Fine-tuning |
|-------|-----------|-------|------|-----------|
| Gemma2 | 8B | Medium | 8-16GB | ❌ Not available |
| Llama3 | 8B | Medium | 10-16GB | ❌ Not available |
| Qwen2 | 8B | Medium | 8-16GB | ❌ Not available |

Use large models if you:
- Have a GPU with 16GB+ VRAM
- Want state-of-the-art embeddings for analysis
- Don't need to fine-tune

## Understanding Your Results

### Evaluation Metrics

The evaluation script computes metrics that show how well embeddings preserve relationships between sensory attributes:

- **Adjacency Matching:** Are directly connected descriptors nearby in embedding space?
- **Distance Matching:** Do embedding distances match the knowledge graph distances?
- **L2 vs Angular:** Different geometric perspectives on embedding quality

Higher scores indicate better preservation of sensory attribute structure.

### Using Embeddings for Analysis

Once you have embeddings, you can:

**Statistical Analysis:**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Load embeddings (pickled numpy arrays)
# Use sklearn for PCA, clustering, correlation analysis
# Use scipy for statistical tests
```

**Visualization:**
- t-SNE: High-dimensional visualization (interactive notebook included)
- PCA: Linear dimensionality reduction
- Clustering: Group similar descriptors (k-means, hierarchical)

**Your Research:**
- Identify sensory attribute clusters
- Analyze descriptor relationships
- Compare models for different food categories
- Extract features for downstream tasks

## Troubleshooting

### "Module not found" Errors

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install -r requirements.txt
python -m pip install --upgrade pip
```

Check dependencies:
```bash
python -c "import torch; import transformers; print('OK')"
```

### Out of Memory (CUDA Error)

```
RuntimeError: CUDA out of memory
```

**Solutions (try in order):**
1. Use a smaller model (BERT or ALBERT)
2. Reduce batch size:
   ```bash
   python3 embeddings_from_pretrained.py --batch_size 2
   ```
3. Use CPU instead:
   ```bash
   python3 embeddings_from_pretrained.py --device cpu
   ```
4. Close other GPU applications

Check GPU status:
```bash
nvidia-smi
```

### Scripts Are Slow

**First run is slower** (downloads models from Hugging Face):
- Verify internet connection
- Check disk space: `df -h` (need 2GB+ free)
- Be patient: First BERT download is ~1.2GB

**Fine-tuning is slow on CPU:**
- GPU is 5-10x faster
- BERT is fastest model
- Large models (T5, BART) take longer

**If stuck/frozen:**
- Check GPU with `nvidia-smi`
- Verify internet isn't interrupted
- Consider using CPU-only mode

### CPU-Only Mode

To use CPU without GPU (slower but works):

```bash
python3 embeddings_from_pretrained.py --device cpu
python3 finetune_BERT_by_sequence_classification.py --device cpu
```

**Expected slowdown:** 5-10x slower than GPU. Fine-tuning overnight is normal on CPU.

### Gemma2/Llama3/Qwen2 Models Not Generating

Large models (8B) are **optional** and disabled by default:

```bash
# Enable large model evaluation
python3 embeddings_from_pretrained.py --enable_LLM
```

Requirements:
- GPU with 16GB+ VRAM
- ~8GB disk space per model
- Internet for first download
- Patience: Inference is slower

## FAQ

**Q: I'm a food researcher with limited computing resources. Where should I start?**

A: Start with BERT (Workflow A). It's fast, accurate, and works on regular computers. You can evaluate it immediately without fine-tuning.

**Q: What's the difference between fine-tuning and using pretrained models?**

A: Pretrained models are already trained on general text, so they understand language well but may not be optimized for your specific sensory descriptors. Fine-tuning adapts the model to your domain, potentially improving embeddings.

**Q: Can I fine-tune Gemma2, Llama3, or Qwen2?**

A: No, this codebase only provides fine-tuning for small models (BERT, RoBERTa, ALBERT, GPT-2, BART, T5). Large models are for evaluation only.

**Q: How long does fine-tuning take?**

A: 30 min - 4 hours on GPU (depending on model).
- BERT: ~30 min
- ALBERT: ~20 min
- RoBERTa/GPT-2: ~40 min
- BART/T5: ~60 min
On CPU, expect 5-10x longer.

**Q: Can I modify the sensory descriptors?**

A: Yes! Edit `generate_finetuned_data.py` in the `src/finetune.py` module to customize the knowledge graph for your food type and descriptors.

**Q: Where are my results saved?**

A: All embeddings and models go to `./outputs/` directory. Evaluation results are printed to console and can be exported to CSV using `--layout_file` option.

**Q: Can I use embeddings from multiple models together?**

A: Yes! Each model produces different embeddings. You can load and combine them for ensemble analysis:
```python
import pickle

# Load embeddings from different models
with open('./outputs/pretrained-BERT_embeddings.pkl', 'rb') as f:
    bert_embeddings = pickle.load(f)

with open('./outputs/pretrained-RoBERTa_embeddings.pkl', 'rb') as f:
    roberta_embeddings = pickle.load(f)

# Combine for ensemble analysis
```

**Q: What if I want to use different food data than the example?**

A: The framework is generalizable to any food category. The knowledge graph structure in `src/graph.py` defines the descriptors and relationships. Modify this to match your domain.

## Files and Structure

**Main Scripts:**
- `embeddings_from_pretrained.py` - Generate pretrained embeddings
- `embeddings_from_finetuned.py` - Generate fine-tuned embeddings
- `models_evaluation.py` - Evaluate and compare embeddings
- `generate_finetuned_data.py` - Create training data
- `finetune_[MODEL]_by_sequence_classification.py` - Fine-tune individual models

**Notebooks:**
- `Notebook_embedding_visualization.ipynb` - Visualize embeddings with t-SNE/PCA
- `Notebook_OOD_visualization.ipynb` - Out-of-distribution analysis

**Source Code:**
- `src/graph.py` - Knowledge graph definition
- `src/models.py` - Language model implementations
- `src/metrics.py` - Evaluation metrics
- `src/finetune.py` - Fine-tuning logic

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

Or in plain text:

```
Chang, Y.-T., & Chen, S.-F. (2026). Sensory-CoKGE: A contextualized knowledge
graph embedding framework using language models for converting text-based food
attributes into numerical representation. Expert Systems with Applications,
299(C), 130191. https://doi.org/10.1016/j.eswa.2025.130191
```
