import os
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple
from jaxtyping import Float, Int, install_import_hook


# Hyperparameters
BATCH_SIZE = 12  # == <BATCH>:  how many independent sequences will we process in parallel?
BLOCK_SIZE = 64  # == <TOKEN>:  what is the maximum context length for predictions?
EMBEDDING_DIM = 128  # == <CHANNEL>:  how many features/dimensions will we use to represent each token?
SEED = 1337
MAX_ITERS = 5000
EVAL_INTERVAL = MAX_ITERS / 10
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 4
DROPOUT = 0.20
NUM_GENERATE_TOKENS = 5000
# ------------
torch.manual_seed(SEED)

# Load input data (currently Shakespeare data from Karpathy's repo)
if not os.path.exists('input.txt'):
    print("downloading data file from github")
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'input.txt')
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique characters that occur in this text
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)  # == <VOCAB>:  the number of unique characters in the text i.e. the vocabulary size

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
ENCODER = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
DECODER = lambda li: ''.join([itos[i] for i in li])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(ENCODER(text), dtype=torch.long)
train_index = int(0.9*len(data))  # first 90% will be train dataset, rest val
train_data = data[:train_index]
val_data = data[train_index:]


def get_batch(split: str) -> Tuple[Int[Tensor, "BATCH TOKEN"], Int[Tensor, "BATCH TOKEN"]]:
    # Generate a small batch of data of inputs x and targets y
    data_to_sample = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_to_sample) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_to_sample[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_to_sample[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss() -> Dict[str, Float[Tensor, ""]]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        calculated_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss_node_i = model(X, Y)
            calculated_losses[k] = loss_node_i.item()
        estimated_split_loss = calculated_losses.mean()
        out[split] = estimated_split_loss
    model.train()
    return out


class GPTLanguageModel(nn.Module):
    """ A GPT language model from scratch """

    def __init__(self):
        super().__init__()
        # Each token directly reads off the embeddings for the next token from lookup table
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        # Positional embeddings are learnable parameters here for simplicity
        self.pos_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[TransformerBlock(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_TRANSFORMER_BLOCKS)])
        self.layer_norm_final = nn.LayerNorm(EMBEDDING_DIM)
        self.linear_model_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(
        self,
        idx: Int[Tensor, "BATCH TOKEN"],
        targets: Optional[Int[Tensor, "BATCH TOKEN"]] = None
    ) -> Tuple[Optional[Float[Tensor, "BATCH TOKEN VOCAB"]], Optional[Float[Tensor, ""]]]:
        B, T = idx.shape

        # Look up token and positional embeddings
        token_embedding = self.token_embedding_table(idx)  # (B, T, C)
        positional_embedding = self.pos_embedding_table(torch.arange(T, device=DEVICE))  # (T, C)

        # Add token and positional embeddings for the input embedding to the transformer stack
        input_embedding = token_embedding + positional_embedding  # (B, T, C) + (T, C) broadcast

        # Pass the input embedding through the transformer stack with a final layer norm
        output_embedding = self.blocks(input_embedding)  # [(B, T, C) -> (B, T, C)] x NUM_TRANSFORMER_BLOCKS
        output_embedding = self.layer_norm_final(output_embedding)  # (B, T, C) @ (C, C) -> (B, T, C)  NOTE: Feed-forward is done separately for each token]:

        # Project back to vocabulary space
        logits = self.linear_model_head(output_embedding)  # (B, T, VOCAB_SIZE)

        if targets is None:
            loss = None
        else:
            B, T, _ = logits.shape
            # Before calculating cross entropy loss, shuffle logits and targets to be 2D
            logits = logits.view(B*T, VOCAB_SIZE)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            logits = None

        return logits, loss

    def generate(self, idx: Int[Tensor, "BATCH TOKEN"], max_new_tokens: int) -> Int[Tensor, "BATCH TOKEN+{max_new_tokens}"]:
        # idx (B, T): array of indices in the current context so far

        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_context = idx[:, -BLOCK_SIZE:]

            # Get the predictions
            logits, _ = self(idx_context)  # (B, T, C)

            # Pluck out the last time step
            logits = logits[:, -1, :]  # (B, T[-1], C) -> (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution stochastically using learned probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T) -> (B, T+1)

        return idx


class Head(nn.Module):
    """ One head of causal self-attention """

    def __init__(self, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        # NOTE: GPT only uses decoder blocks (i.e. using causal self-attention) so we don't allow the model to learn from future tokens at each timestep
        # The tril variable allows us to do exactly that by masking future attention scores.
        # We also make it a buffer since it's fixed has no learnable parameters
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: Float[Tensor, "BATCH TOKEN CHANNEL"]) -> Float[Tensor, "BATCH TOKEN {self.head_size}"]:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, HS)
        q = self.query(x)  # (B, T, HS)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, HS) @ (B, HS, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, HS)
        out = wei @ v  # (B, T, T) @ (B, T, HS) -> (B, T, HS)

        return out


class MultiHead(nn.Module):
    """ Multiple heads of causal self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: Float[Tensor, "BATCH TOKEN CHANNEL"]) -> Float[Tensor, "BATCH TOKEN CHANNEL"]:
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, HS * num_heads = C)
        out = self.dropout(self.projection(out))  # (B, T, C) -> (B, T, C)
        return out
    

class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x: Float[Tensor, "BATCH TOKEN CHANNEL"]) -> Float[Tensor, "BATCH TOKEN CHANNEL"]:
        return self.network(x)
    

class TransformerBlock(nn.Module):
    """ A single Transformer block: communication followed by computation """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)  # NOTE: per batch, per token normalization
        self.self_attention = MultiHead(num_heads, head_size)

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x: Float[Tensor, "BATCH TOKEN CHANNEL"]) -> Float[Tensor, "BATCH TOKEN CHANNEL"]:
        # We add to x itself to do the "residual" or "skip connection"
        # NOTE: This lets us propagate gradients due to supervision all the way to the early part of the network
        # GRADIENT SUPERHIGHWAY
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# Enable runtime type checking on all functions
with install_import_hook("gpt", "beartype.beartype"):
    from gpt import *

# Initialize the model
model = GPTLanguageModel()
m = model.to(DEVICE)
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
iteration = 0
try:
    for iteration in range(MAX_ITERS):
        # Every once in a while evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = estimate_loss()
            print(f"step {iteration}/{MAX_ITERS}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        _, loss_node = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss_node.backward()
        optimizer.step()

except KeyboardInterrupt:
    print(f"Training was manually killed at iteration: {iteration}")
    pass

# Generate from the model
starting_empty_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(DECODER(m.generate(starting_empty_context, max_new_tokens=NUM_GENERATE_TOKENS)[0].tolist()))
