import torch

from pathlib import Path

from cs336_basics.bpe import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizers import AdamW


input_path = Path("tests/fixtures/tinystories_sample_5M.txt")
checkpoint_path = Path("experiments/tiny_checkpoint.pt")
tokenizer_path = Path("experiments/tiny_tokenizer.pt")
vocab_size = 1000
context_length = 64
d_model = 128
num_layers = 2
num_heads = 4
d_ff = 512
theta = 10000.0
special_tokens = ["<|endoftext|>"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer_state = torch.load(
    tokenizer_path,
    weights_only=False,
)

tokenizer = Tokenizer(
    tokenizer_state["vocab"],
    tokenizer_state["merges"],
    tokenizer_state["special_tokens"],
)

model = TransformerLM(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    theta,
).to(device)

optimizer = AdamW(model.parameters())

iteration = load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    )

model.eval()
print(f"loaded checkpoint from step {iteration}")

prompt = input("Prompt: ")
max_new_tokens = 100

generated_ids = tokenizer.encode(prompt)

with torch.no_grad():
    for _ in range(max_new_tokens):
        context_ids = generated_ids[-context_length:]
        input_ids = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        
        logits = model(input_ids)
        last_logits = logits[0, -1]

        probabilities = softmax(last_logits, dim=-1)

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
        ).item()

        generated_ids.append(next_token_id)

generated_text = tokenizer.decode(generated_ids)

print(generated_text)


