import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
  def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
    super().__init__()

    assert d_out % num_heads == 0, "d_out must be divisible by num_head"
    self.d_in = d_in
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim=d_out//num_heads # # Reduce the projection dim to match desired output dim

    self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.out_proj = nn.Linear(d_out,d_out) # # Linear layer to combine head outputs
    self.dropout=nn.Dropout(dropout)
    self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))

  def forward(self,x):

    b,num_tokens,d_in=x.shape

    keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
    queries = self.W_query(x)
    values = self.W_value(x)

    # We implicitly split the matrix by adding a `num_heads` dimension
    # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
    queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
    values = values.view(b,num_tokens,self.num_heads,self.head_dim)

    # Transpose : (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1,2)
    queries = queries.transpose(1,2)
    values = values.transpose(1,2)

    # Compute scaled dot-product attention with casual mask
    attn_scores = queries @ keys.transpose(2,3) # output shape is (b,num_head,num_tokens ,num_tokens)
    #(b, num_heads,num_tokens, head_dim)(b, num_heads,head_dim,num_tokens)
                              #                          ^-------^     changes for matrix multiplication

    # Original mask truncated to the number of tokens and converted to boolean
    mask_bool=self.mask.bool()[:num_tokens,:num_tokens] # basic indexing and slicing

    # USe the mask to fill attention scores
    attn_scores.masked_fill_(mask_bool,-torch.inf)

    attn_weight=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
    attn_weight=self.dropout(attn_weight)

    # Shape:(b,num_tokens,num_heads,head_dim)
    context_vec=(attn_weight @ values).transpose(1,2)
    # (b,num_head,num_tokens ,num_tokens) @ (b, num_heads, num_tokens, head_dim) -> (b, num_heads, num_tokens, head_dim).transpose(1,2)= (b,num_tokens,num_heads, head_dim)

    # Combine head, where d_out=self.num_heads*self.head_dim
    context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
    # Contiguous memory means the tensor’s elements are stored next to each other in a single continuous block of memory in the order PyTorch expects.
    context_vec = self.out_proj(context_vec)
    # context_vec=self.out_proj(context_vec) means [output_vector = input_vector × W + b]

    return context_vec

class LayerNorm(nn.Module):
  def __init__(self,emb_dim):
    super().__init__()
    self.eps=1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self,x):
    mean = x.mean(dim=-1,keepdim=True)
    var = x.var(dim=-1,keepdim=True,unbiased=False)
    norm_x = (x-mean)/torch.sqrt(var + self.eps)
    return norm_x*self.scale+self.shift
  
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):

    return 0.5 * x *(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715 * torch.pow(x,3))))


class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
        GELU(),
        nn.Linear(4* cfg["emb_dim"],cfg["emb_dim"])
    )

  def forward(self,x):
    return self.layers(x)

class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.att = MultiHeadAttention(
        d_in = cfg["emb_dim"],
        d_out = cfg["emb_dim"],
        context_length = cfg["context_length"],
        num_heads = cfg["n_heads"],
        dropout = cfg["drop_rate"],
        qkv_bias= cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self,x):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x) # Shape [batch_size, num_tokens, emb_size]
    x = self.drop_shortcut(x)
    x = x + shortcut # Add the original input back

    # Shortcut connection for feed-forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut # Add the original input back

    return x
  
class GPTModel(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias = False)
    self.out_head.weight = self.tok_emb.weight  # weight tying

  def forward(self,in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))

    x = tok_embeds + pos_embeds # Shape [batch_size, num_tokens, emb_size]
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)

    return logits


def generate_text(model,idx,max_new_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):

  for _ in range(max_new_tokens):

    idx_cond = idx[:,-context_size:]

    # Get the prediction
    with torch.no_grad():
      logits = model(idx_cond)

    # Focus only on the last time step
    # (batch, n_token, vocab_size) becomes (batch, vocab_size)
    logits = logits[: , -1, :]

    # New: Filter logits with top_k sampling
    if top_k is not None:
      # keep only top_k values
      top_logits,_ = torch.topk(logits,top_k)
      min_val = top_logits[:, -1] # all rows and only last column that is small number
      logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device),logits)

     # New: Apply temperature scaling
    if temperature > 0.0:
      logits =   logits / temperature

      logits = logits - logits.max(dim = -1,keepdim= True).values

      # Apply softmax to get probabilities
      probs = torch.softmax(logits, dim=-1) #shape (batch_size,context_length)

      # sample from the distribution
      idx_next = torch.multinomial(probs,num_samples=1) #(batch_size,1)

    # Otherwise same as before: get idx of the vocab entry with the highest logits value
    else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

    if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        break

    # Append sampled index to the running sequence
    idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

  return idx

def text_to_token_ids(text,tokenizer):
  encoded = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
  return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
  flat = token_ids.squeeze(0) # remove batch dimension
  return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch,target_batch,model,device):
  input_batch,target_batch = input_batch.to(device),target_batch.to(device)
  logits = model(input_batch)
  loss = nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten()) # Here cross_entropy is function calling that why we use nn.functional.___
  return loss

#Instead of checking model performance on just one batch of sentences, we check it on many batches and compute the average loss.
def calc_loss_loader(data_loader,model,device,num_batches = None):
  total_loss = 0

  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches,len(data_loader))

  for i,(input_batch,target_batch) in enumerate(data_loader):
    if i < num_batches:
      loss = calc_loss_batch(input_batch,target_batch,model,device)
      total_loss += loss.item()
    else:
      break

  return total_loss/num_batches

import pytorch_lightning as pl
import torch

class LitLanguageModel(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs,
        tokenizer,
        train_loader,
        val_loader,
        eval_freq,
        eval_iter,
        start_context
    ):
        super().__init__()

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.tokenizer = tokenizer
        self.train_loader_ref = train_loader
        self.val_loader_ref = val_loader

        self.eval_freq = eval_freq # Run evaluation every N steps
        self.eval_iter = eval_iter # Number of batches used to estimate loss
        self.start_context = start_context # Prompt used to generate sample text

        self.tokens_seen = 0 # Helps measure training scale.
        self.global_step_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.track_tokens_seen = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        input_batch, target_batch = batch

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            self.model,
            self.device
        )

        self.tokens_seen += input_batch.numel() # count how many tokens the model has processed so far.
        self.global_step_counter += 1 # Each batch processed = one optimization step.

        self.log("train_loss_step", loss, prog_bar=True) # Lightning logging API

        # evaluation logic
        if self.global_step_counter % self.eval_freq == 0:

            train_loss, val_loss = self.evaluate_model()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.track_tokens_seen.append(self.tokens_seen)

            print(
                  f"Ep {self.current_epoch+1} "
                  f"(Step {self.global_step_counter}) "
                  f"Train loss {round(train_loss, 4)}, "
                  f"Val loss {round(val_loss, 4)}"
              )

            self.log("train_loss_eval", train_loss, prog_bar=True)
            self.log("val_loss_eval", val_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        loss = calc_loss_batch(
            input_batch, target_batch,
            self.model, self.device
        )
        self.log("val_loss_step", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.generate_and_print_sample()

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(
                self.train_loader_ref, self.model,
                self.device, num_batches=self.eval_iter
            )
            val_loss = calc_loss_loader(
                self.val_loader_ref, self.model,
                self.device, num_batches=self.eval_iter
            )
        self.model.train()
        return train_loss, val_loss

    def generate_and_print_sample(self):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0] # (max_sequence_length, embedding_dimension)
        encoded = text_to_token_ids(
            self.start_context, self.tokenizer
        ).to(self.device)

        with torch.no_grad():
            token_ids = generate_text(
                model=self.model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size
            )

        decoded_text = token_ids_to_text(token_ids, self.tokenizer)
        print(f"\n[Sample] {decoded_text.replace(chr(10), ' ')}\n")
        self.model.train()

    def configure_optimizers(self):
        return self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
