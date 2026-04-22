# TinyStories GPT-2 (124M) From Scratch

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model%20Card-ffcc00.svg)](https://hf.co/snehangshu511/tinystories-gpt2-124M-scratch)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

A 124M-parameter GPT-2 style language model built entirely from scratch in raw PyTorch and trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 

This project was built as a hands-on deep dive into the architecture of Large Language Models. It features a completely custom, from-scratch implementation of the Transformer decoder block, integrated seamlessly into the Hugging Face `transformers` ecosystem.

**[Try out the model on Hugging Face!](https://hf.co/snehangshu511/tinystories-gpt2-124M-scratch)**

---

## Model Details

| | |
|---|---|
| **Architecture** | Decoder-only Transformer (GPT-2 style) |
| **Parameters** | ~124M |
| **Layers** | 12 Transformer blocks |
| **Attention heads** | 12 |
| **Embedding dim** | 768 |
| **Context length** | 128 tokens |
| **Vocabulary** | 50,257 (GPT-2 BPE via `tiktoken`) |
| **Training data** | 20,000 TinyStories samples |
| **Validation data** | 5,000 TinyStories samples |

---

## Repository Structure

- `train_tinystories_gpt2.ipynb`: The main Google Colab / Jupyter notebook containing the from-scratch PyTorch architecture, training loop (via PyTorch Lightning), and Hugging Face export logic.
- `hf_model.py`: The custom Python wrapper (`SmallLLMForCausalLM`) that allows the raw PyTorch model to be fully compatible with Hugging Face's `AutoModel` and `pipeline` features.
- `pyproject.toml`: The modern Python package configuration detailing all dependencies required to run the code.
- `tinystories-gpt2-124M/`: The output folder containing the exported Hugging Face weights, tokenizer, and config (ignored by git, hosted on HF).

---

## How to Use (Inference)

Because this model uses a custom architecture, you **must** pass `trust_remote_code=True` when loading the model via Hugging Face.

First, install the required packages:
```bash
pip install transformers torch
```

### Option 1: Using the Hugging Face Pipeline (Easiest)

```python
import torch
from transformers import pipeline

# Set up the text generation pipeline
generator = pipeline(
    "text-generation",
    model="snehangshu511/tinystories-gpt2-124M-scratch",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate a story!
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

### Option 2: Manual Loading with AutoModelForCausalLM

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

repo_id = "snehangshu511/tinystories-gpt2-124M-scratch"

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare input
inputs = tokenizer("A boy named Max had a big red ball", return_tensors="pt").to(device)

# Configure Generation
gen_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id
)

# Generate Text
with torch.no_grad():
    output_ids = model.generate(inputs.input_ids, generation_config=gen_config)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Limitations

This model is trained specifically on the **TinyStories** dataset. 
- It excels at generating very simple, childlike stories with basic vocabulary.
- It is **not** an instruction-following model or a chatbot.
- It struggles with complex logic, historical facts, or technical text.

---

## Author

**Snehangshu Bhuin**
- GitHub: [snehangshu2002](https://github.com/snehangshu2002)
- Hugging Face: [snehangshu511](https://huggingface.co/snehangshu511)
