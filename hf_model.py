"""
hf_model.py
===========
Wraps the custom GPTModel (defined in the notebook) so it is fully compatible
with the Hugging Face Transformers library.

Key design decisions
--------------------
* We inherit from `PreTrainedModel` so that `save_pretrained()` /
  `from_pretrained()` work out of the box.
* The architecture stays **identical** to what was trained in the notebook —
  no weight surgery needed as long as the parameter names match.
* We re-use `GPT2Config` for the config object; our custom field names
  (`emb_dim`, `n_heads`, `n_layers`, `drop_rate`) are stored as extra
  `kwargs` in the config so the config file remains valid JSON.
* The weight-key mapping between the custom `GPTModel` and the standard
  HF `GPT2LMHeadModel` format is handled in `convert_to_hf.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# 1.  Model building-blocks  (identical to notebook, no changes)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out      = d_out
        self.num_heads  = num_heads
        self.head_dim   = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool   = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


class GELU(nn.Module):
    def forward(self, x):
        import math
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in           = cfg["emb_dim"],
            d_out          = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout        = cfg["drop_rate"],
            num_heads      = cfg["n_heads"],
            qkv_bias       = cfg["qkv_bias"],
        )
        self.ff       = FeedForward(cfg)
        self.norm1    = LayerNorm(cfg["emb_dim"])
        self.norm2    = LayerNorm(cfg["emb_dim"])
        self.drop_res = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_res(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_res(x)
        return x + shortcut


class _GPTModelCore(nn.Module):
    """The raw GPTModel from the notebook (unchanged logic)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


# ---------------------------------------------------------------------------
# 2.  HF-compatible wrapper
# ---------------------------------------------------------------------------

class SmallLLMConfig(GPT2Config):
    """
    Extends GPT2Config with the extra fields used by the notebook config dict.
    Stored fields are forwarded to the parent so the JSON round-trip works.
    """
    model_type = "small_llm_gpt2"

    def __init__(self,
                 emb_dim: int = 768,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 drop_rate: float = 0.1,
                 qkv_bias: bool = False,
                 context_length: int = 128,
                 **kwargs):
        # Map our names → GPT2Config canonical names so HF tooling works
        kwargs.setdefault("n_embd",      emb_dim)
        kwargs.setdefault("n_head",      n_heads)
        kwargs.setdefault("n_layer",     n_layers)
        kwargs.setdefault("n_positions", context_length)
        kwargs.setdefault("n_ctx",       context_length)
        kwargs.setdefault("resid_pdrop", drop_rate)
        kwargs.setdefault("embd_pdrop",  drop_rate)
        kwargs.setdefault("attn_pdrop",  drop_rate)
        super().__init__(**kwargs)
        # Store our custom fields too so they survive save/load
        self.emb_dim        = emb_dim
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.drop_rate      = drop_rate
        self.qkv_bias       = qkv_bias
        self.context_length = context_length

    def to_notebook_cfg(self) -> dict:
        """Return the dict format used by the notebook."""
        return {
            "vocab_size":     self.vocab_size,
            "context_length": self.context_length,
            "emb_dim":        self.emb_dim,
            "n_heads":        self.n_heads,
            "n_layers":       self.n_layers,
            "drop_rate":      self.drop_rate,
            "qkv_bias":       self.qkv_bias,
        }


class SmallLLMForCausalLM(PreTrainedModel):
    """
    HuggingFace-compatible GPT-2 style causal language model.

    Usage after training
    --------------------
    # Save
    model_hf = SmallLLMForCausalLM(config)
    model_hf.gpt.load_state_dict(trained_gpt_model.state_dict())
    model_hf.save_pretrained("tinystories-gpt2-124M")

    # Load back
    model_hf = SmallLLMForCausalLM.from_pretrained("tinystories-gpt2-124M")

    # Use with AutoModel
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.register(SmallLLMConfig, SmallLLMForCausalLM)
    model = AutoModelForCausalLM.from_pretrained("tinystories-gpt2-124M")
    """

    config_class = SmallLLMConfig
    base_model_prefix = "gpt"

    def __init__(self, config: SmallLLMConfig):
        super().__init__(config)
        self.gpt = _GPTModelCore(config.to_notebook_cfg())
        self.post_init()

    # ------------------------------------------------------------------
    # Forward — returns HF-standard CausalLMOutputWithPast so pipelines work
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      Optional[torch.LongTensor]   = None,
        attention_mask: Optional[torch.FloatTensor]  = None,
        labels:         Optional[torch.LongTensor]   = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        logits = self.gpt(input_ids)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss   = loss,
            logits = logits,
        )

    # Needed for text-generation pipeline
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def get_output_embeddings(self):
        return self.gpt.out_head

    def set_output_embeddings(self, new_embeddings):
        self.gpt.out_head = new_embeddings
