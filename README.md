<div align="center">

# TinyStories GPT-2 (124M) — Built From Scratch

<p align="center">
  <a href="https://hf.co/snehangshu511/tinystories-gpt2-124M-scratch">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Card-ffcc00?style=for-the-badge" alt="Hugging Face Model"/>
  </a>
  <a href="https://snehangshu511-tinystories-gpt2-demo.hf.space/">
    <img src="https://img.shields.io/badge/Live%20Demo-Spaces-blue?style=for-the-badge" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white" alt="PyTorch Lightning"/>
  <img src="https://img.shields.io/badge/Parameters-124M-green?style=for-the-badge" alt="124M Parameters"/>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/snehangshu2002/tinystories-gpt2-from-scratch/main/img/Screenshot%202026-04-22%20202232.png" alt="Demo Preview" width="800"/>
</p>
<p align="center">
  A <strong>124M-parameter GPT-2 style language model</strong> implemented entirely from scratch in raw PyTorch,<br/>
  trained on the <a href="https://huggingface.co/datasets/roneneldan/TinyStories">TinyStories</a> dataset, and fully integrated into the Hugging Face ecosystem.
</p>

**Live Demo:** [snehangshu511-tinystories-gpt2-demo.hf.space](https://snehangshu511-tinystories-gpt2-demo.hf.space/)

</div>

---

## About

This project is a hands-on deep dive into building Large Language Models from the ground up. Every component — from the multi-head self-attention mechanism to the positional embeddings — is implemented in **raw PyTorch**, without relying on high-level transformer libraries.

The model is trained to generate short, coherent children's stories and is fully compatible with the Hugging Face `transformers` ecosystem via a custom model wrapper.

**Key highlights:**
- From-scratch Transformer decoder — every layer built in raw PyTorch
- Hugging Face compatible — works with `AutoModel`, `pipeline`, and `AutoTokenizer`
- PyTorch Lightning training loop — clean, scalable, and reproducible
- Custom `trust_remote_code` integration, seamlessly hosted on Hugging Face Hub

---

## Model Architecture

| Parameter | Value |
|---|---|
| **Architecture** | Decoder-only Transformer (GPT-2 style) |
| **Total Parameters** | ~124M |
| **Transformer Layers** | 12 |
| **Attention Heads** | 12 |
| **Embedding Dimension** | 768 |
| **Context Length** | 128 tokens |
| **Vocabulary Size** | 50,257 (GPT-2 BPE via `tiktoken`) |
| **Training Samples** | 20,000 TinyStories stories |
| **Validation Samples** | 5,000 TinyStories stories |

---

## Repository Structure

```
.
├── .gitignore                  # Ignore unnecessary files
├── README.md                  # Project documentation
├── app.py                     # Inference / app entry point
├── hf_model.py                # Hugging Face model integration
├── pyproject.toml             # Project dependencies & config
├── train_tinystories_gpt2.ipynb  # Training notebook (GPT-2 from scratch)
└── uv.lock                    # Dependency lock file
```

| File | Description |
|---|---|
| `train_tinystories_gpt2.ipynb` | Full GPT-2 architecture, PyTorch Lightning training loop, and Hugging Face export logic |
| `hf_model.py` | `SmallLLMForCausalLM` wrapper enabling `AutoModel` and `pipeline` compatibility |
| `pyproject.toml` | All Python dependencies for reproducibility |

---

## Quick Start

### Installation

```bash
pip install transformers torch
```

> **Note:** This model uses a custom architecture. Always pass `trust_remote_code=True` when loading.

### Option 1 — Hugging Face Pipeline (Recommended)

```python
import torch
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="snehangshu511/tinystories-gpt2-124M-scratch",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

output = generator(
    "Tom and his friend were playing",
    max_new_tokens=100,
    max_length=None,
    pad_token_id=generator.tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.8,
    top_k=50
)

print(output[0]['generated_text'])
```

### Option 2 — Manual Loading with AutoModelForCausalLM

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

repo_id = "snehangshu511/tinystories-gpt2-124M-scratch"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Tokenize input
inputs = tokenizer("A boy named Max had a big red ball", return_tensors="pt").to(device)

# Configure generation
gen_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id
)

# Run inference
with torch.no_grad():
    output_ids = model.generate(inputs.input_ids, generation_config=gen_config)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Limitations

This model is purpose-built for the TinyStories domain and has the following constraints:

- Excels at generating simple, childlike stories with basic vocabulary
- Not an instruction-following model or chatbot
- Struggles with complex reasoning, factual knowledge, or technical content
- Limited context window of 128 tokens

---

## Links

| Resource | Link |
|---|---|
| Model on Hugging Face | [snehangshu511/tinystories-gpt2-124M-scratch](https://hf.co/snehangshu511/tinystories-gpt2-124M-scratch) |
| Live Interactive Demo | [snehangshu511-tinystories-gpt2-demo.hf.space](https://snehangshu511-tinystories-gpt2-demo.hf.space/) |
| Training Dataset | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |

---

## Author

**Snehangshu Bhuin**

[![GitHub](https://img.shields.io/badge/GitHub-snehangshu2002-181717?style=flat&logo=github)](https://github.com/snehangshu2002)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-snehangshu511-ffcc00?style=flat)](https://huggingface.co/snehangshu511)
