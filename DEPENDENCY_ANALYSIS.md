# Dependency Analysis - sensory_cokge

## Overview
This document provides a comprehensive analysis of all function dependencies in the codebase, organized by module and dependency type.

---

## External Library Dependencies

### Required Dependencies
These are core dependencies needed for the package to function:

| Library | Usage | Required For |
|---------|-------|--------------|
| **numpy** | Numerical operations, matrices | Graph properties, metrics, normalization |
| **torch** | Deep learning, tensor operations | All embedding functions, distance calculations |
| **pandas** | Data manipulation, CSV I/O | Helper functions, data export |
| **igraph** | Graph data structures | DescriptionGraph, graph operations |
| **transformers** | Pretrained language models | All model embedding functions (except Laser) |
| **scikit-learn** | Machine learning utilities | (Used in evaluation scripts) |
| **matplotlib** | Visualization | Plotting functions |
| **datasets** | Hugging Face datasets | Fine-tuning scripts |
| **sentencepiece** | Tokenization | Required by some transformers models |
| **protobuf** | Serialization | Required by transformers |
| **opentsne** | t-SNE visualization | Dimensionality reduction |
| **adjustText** | Plot text adjustment | Visualization |
| **tensorboardX** | Training visualization | Fine-tuning monitoring |

### Optional Dependencies
These dependencies are only needed for specific features:

| Library | Usage | When Needed |
|---------|-------|-------------|
| **laser_encoders** | LASER embeddings | Only for `Laser_embeddings()` function - lazy imported |

### Standard Library Dependencies

| Library | Usage |
|---------|-------|
| **os** | File operations, path handling |
| **json** | JSON serialization |
| **pickle** | Object serialization |
| **yaml** | Configuration parsing |
| **copy** | Deep copying objects |
| **random** | Random sampling |
| **hashlib** | Hash generation for synthetic data |

---

## Module-Level Dependency Analysis

### 1. `sensory_cokge/models.py`

**Module Imports:**
- `os` - standard library
- `torch` - required
- `torch.utils.data` (DataLoader, Dataset) - required
- `transformers` (multiple tokenizers and models) - required
- `.utils.use_a_or_an` - internal

**Functions and Their Dependencies:**

#### Core Embedding Functions

| Function | Required Libraries | Model-Specific |
|----------|-------------------|----------------|
| `ALBERT_embeddings()` | torch, transformers (AlbertTokenizer, AlbertModel) | ALBERT |
| `BERT_embeddings()` | torch, transformers (BertTokenizer, BertModel) | BERT |
| `BART_embeddings()` | torch, transformers (BartTokenizer, BartModel) | BART |
| `Gemma2_embeddings()` | torch, transformers (AutoTokenizer, Gemma2Model) | Gemma2 |
| `GPT2_embeddings()` | torch, transformers (GPT2Tokenizer, GPT2Model) | GPT2 |
| `Llama3_embeddings()` | torch, transformers (AutoTokenizer, LlamaModel) | Llama3 |
| `Qwen2_embeddings()` | torch, transformers (Qwen2Tokenizer, Qwen2Model) | Qwen2 |
| `RoBERTa_embeddings()` | torch, transformers (RobertaTokenizer, RobertaModel) | RoBERTa |
| `T5_embeddings()` | torch, transformers (T5Tokenizer, T5Model) | T5 |
| `Laser_embeddings()` | **laser_encoders** (LaserEncoderPipeline) - LAZY IMPORT | LASER |

**Special Notes:**
- `Laser_embeddings()` uses **lazy import** - laser_encoders is only imported when the function is called
- If laser_encoders is not installed, calling `Laser_embeddings()` raises an informative ImportError
- All other embedding functions require transformers and torch at import time

#### Helper Classes

| Class/Function | Dependencies |
|----------------|--------------|
| `TextDataset` | torch.utils.data.Dataset |

---

### 2. `sensory_cokge/graph.py`

**Module Imports:**
- `os` - standard library
- `copy` - standard library
- `numpy` - required
- `igraph` (Graph, plot) - required

**Functions and Their Dependencies:**

| Function/Class | Required Libraries | Purpose |
|----------------|-------------------|---------|
| `DescriptionGraph` | igraph, numpy, copy | Graph structure for descriptions |
| `description_graph()` | igraph, numpy | Create predefined coffee flavor wheel |
| `graph_properties()` | igraph, numpy | Extract graph metrics |
| `normalize_weighted_adjacency()` | numpy | Normalize adjacency matrices |
| `coffee_flavor_wheel_descriptions()` | (internal logic only) | Generate coffee graph structure |

---

### 3. `sensory_cokge/utils.py`

**Module Imports:**
- `os` - standard library
- `yaml` - required
- `json` - standard library
- `pickle` - standard library

**Functions and Their Dependencies:**

| Function | Required Libraries | Purpose |
|----------|-------------------|---------|
| `init_directory()` | os | Create directories |
| `save_object()` | pickle, json | Serialize objects |
| `load_pickle_obj()` | pickle | Load pickle files |
| `load_json_obj()` | json | Load JSON files |
| `use_a_or_an()` | (none - pure logic) | Grammar helper |
| `parse_training_args()` | yaml, os | Parse YAML configs |

**Note:** `yaml` is currently used but not in requirements.txt - should be added!

---

### 4. `sensory_cokge/finetune.py`

**Module Imports:**
- `random` - standard library
- `hashlib` - standard library
- `.utils.use_a_or_an` - internal
- `.graph.description_graph, NOT_COUNT_DESCRIPTIONS` - internal

**Functions and Their Dependencies:**

| Function | Required Libraries | Purpose |
|----------|-------------------|---------|
| `generate_finetune_data()` | random, hashlib, graph module | Generate synthetic training data |
| `_construct_single_data()` | random, graph module | Build single training sample |
| `_simple_hash()` | hashlib | Hash sample combinations |

---

### 5. `sensory_cokge/metrics.py`

**Module Imports:**
- `torch` - required
- `numpy` - required
- `.graph` (normalize_weighted_adjacency, CONNENTION_DISTANCES) - internal
- `.distances` (angle_differences, l2_differences) - internal

**Functions and Their Dependencies:**

| Class/Method | Required Libraries | Purpose |
|--------------|-------------------|---------|
| `EvaluationMetrics` | torch, numpy, distances module | Evaluate embedding quality |

---

### 6. `sensory_cokge/distances.py`

**Module Imports:**
- `torch` - required

**Functions and Their Dependencies:**

| Function | Required Libraries | Purpose |
|----------|-------------------|---------|
| `cosine_similarity()` | torch | Compute cosine similarity |
| `angle_differences()` | torch | Compute angular distance |
| `l2_differences()` | torch | Compute L2 distance |

---

### 7. `sensory_cokge/relative_embedding.py`

**Module Imports:**
- `torch` - required
- `.distances.cosine_similarity` - internal

**Functions and Their Dependencies:**

| Function/Class | Required Libraries | Purpose |
|----------------|-------------------|---------|
| `construct_relative_embeddings()` | torch, distances module | Project to anchor space |
| `RelativeEmbedding` | torch, distances module | Relative embedding projector |

---

### 8. `sensory_cokge/__init__.py`

**Module Imports:**
- `os` - standard library
- `pandas` - required
- `torch` - required
- All internal modules

**Helper Functions and Their Dependencies:**

| Function | Required Libraries | Purpose |
|----------|-------------------|---------|
| `compute_embeddings()` | torch, all model functions | Unified embedding interface |
| `embeddings_to_csv()` | pandas, torch | Export embeddings to CSV |
| `evaluate_embeddings()` | metrics, graph modules | Evaluate embedding quality |
| `generate_synthetic_data()` | finetune, graph modules | Generate training data |
| `create_description_graph()` | graph module | Build custom graphs |
| `build_graph_from_hierarchy()` | graph module | Build from nested dict |
| `build_graph_from_csv()` | pandas, graph module | Build from CSV file |
| `create_context_template()` | (none - string formatting) | Create context strings |
| `validate_graph_structure()` | graph module | Validate graph structure |

---

## Dependency Tree Summary

```
sensory_cokge/
├── Core Dependencies (Always Required)
│   ├── numpy - numerical operations
│   ├── torch - deep learning framework
│   ├── pandas - data manipulation
│   ├── igraph - graph structures
│   └── transformers - language models
│
├── Visualization Dependencies
│   ├── matplotlib - plotting
│   ├── opentsne - t-SNE
│   └── adjustText - plot labels
│
├── Training Dependencies
│   ├── datasets - HuggingFace datasets
│   ├── tensorboardX - training monitoring
│   ├── sentencepiece - tokenization
│   └── protobuf - serialization
│
├── Optional Dependencies
│   └── laser_encoders - LASER model (lazy import)
│
└── Missing from requirements.txt
    └── yaml - configuration parsing (NEEDS TO BE ADDED!)
```

---

## Function Dependency Matrix

### Embedding Functions

| Function | torch | transformers | laser_encoders | Comments |
|----------|-------|--------------|----------------|----------|
| ALBERT_embeddings | ✓ | ✓ | ✗ | |
| BERT_embeddings | ✓ | ✓ | ✗ | |
| BART_embeddings | ✓ | ✓ | ✗ | |
| Gemma2_embeddings | ✓ | ✓ | ✗ | |
| GPT2_embeddings | ✓ | ✓ | ✗ | |
| Llama3_embeddings | ✓ | ✓ | ✗ | |
| Qwen2_embeddings | ✓ | ✓ | ✗ | |
| RoBERTa_embeddings | ✓ | ✓ | ✗ | |
| T5_embeddings | ✓ | ✓ | ✗ | |
| Laser_embeddings | ✗ | ✗ | ✓ (lazy) | **Optional dependency** |

### Core Helper Functions

| Function | torch | pandas | numpy | igraph | Comments |
|----------|-------|--------|-------|--------|----------|
| compute_embeddings | ✓ | ✗ | ✗ | ✗ | Calls embedding functions |
| embeddings_to_csv | ✓ | ✓ | ✗ | ✗ | CSV export |
| evaluate_embeddings | ✓ | ✗ | ✓ | ✓ | Uses metrics module |
| generate_synthetic_data | ✗ | ✗ | ✗ | ✓ | Uses finetune module |
| create_description_graph | ✗ | ✗ | ✓ | ✓ | Graph construction |
| build_graph_from_hierarchy | ✗ | ✗ | ✓ | ✓ | Graph from dict |
| build_graph_from_csv | ✗ | ✓ | ✓ | ✓ | Graph from CSV |

---

## Recommendations

### 1. Missing Dependencies
Add to `requirements.txt`:
```
pyyaml  # Currently used in utils.py but not in requirements
```

### 2. Optional Dependencies Section
Consider creating a `requirements-optional.txt`:
```
laser_encoders  # For Laser_embeddings() only
```

### 3. Development Dependencies
Consider creating a `requirements-dev.txt` for development tools:
```
pytest
black
flake8
mypy
```

### 4. Dependency Groups
For modern package management (pyproject.toml):
```toml
[project.optional-dependencies]
laser = ["laser_encoders"]
dev = ["pytest", "black", "flake8"]
all = ["laser_encoders"]
```

---

## Import-Time vs Runtime Dependencies

### Import-Time (Required at module import)
- All libraries except `laser_encoders`
- These are checked when doing `import sensory_cokge`

### Runtime (Required only when calling specific functions)
- `laser_encoders` - only needed when calling `Laser_embeddings()`
- This allows the package to work without laser_encoders installed

---

## Conclusion

The codebase has been successfully refactored to make `laser_encoders` an optional dependency through lazy importing. The only remaining issue is that `yaml` (or `pyyaml`) should be added to requirements.txt as it's currently used in `utils.py` but not declared as a dependency.

**Current Status:**
- ✅ laser_encoders is now optional (lazy import implemented)
- ✅ All core functions work without laser_encoders
- ⚠️ yaml/pyyaml needs to be added to requirements.txt
- ✅ All other dependencies are properly declared
