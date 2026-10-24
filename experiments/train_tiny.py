import torch

import numpy as np
from pathlib import Path

from cs336_basics.bpe import train_bpe, Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_grad_norm_
from cs336_basics.optimizers import AdamW, get_lr_cosine_schedule
from cs336_basics.checkpoint import save_checkpoint

input_path = Path("tests/fixtures/tinystories_sample_5M.txt")
checkpoint_path = Path("experiments/tiny_checkpoint.pt")
tokenizer_path = Path("experiments/tiny_tokenizer.pt")
vocab_size = 1000
batch_size = 16
context_length = 64
d_model = 128
num_layers = 2
num_heads = 4
d_ff = 512
theta = 10000.0
max_l2_norm = 1.0
train_steps = 200
special_tokens = ["<|endoftext|>"]


vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

tokenizer = Tokenizer(vocab, merges, special_tokens)
text = input_path.read_text(encoding="utf-8")
ids = tokenizer.encode(text)
##print(len(ids), ids[:20])

token_ids = np.array(ids, dtype=np.int64)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta)
model = model.to(device)
learning_rate = 1e-3

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
)

for step in range(train_steps):
    x, y = get_batch(token_ids, batch_size, context_length, device)
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_l2_norm)
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

save_checkpoint(
    model,
    optimizer,
    train_steps,
    checkpoint_path,
)
torch.save(
    {
        "vocab": vocab,
        "merges": merges,
        "special_tokens": special_tokens,
    },
    tokenizer_path,
)



