"""
Train the tiny GPT on a text file, character by character.

Run with:  python train.py

This will train for a few minutes on a laptop CPU and then generate a
sample of new text in the trained model's "style". Edit the settings
in CONFIG below to make it bigger/smaller/faster.
"""

import os
import time
import torch
from model import GPT

# --------------------------------------------------------------------------
# CONFIG — tweak these. Bigger numbers = better text but slower training.
# --------------------------------------------------------------------------
CONFIG = {
    "data_file": "input.txt",
    "block_size": 128,      # how many characters of context the model sees
    "n_embd": 128,          # width of the model
    "n_head": 4,            # number of attention heads
    "n_layer": 4,           # number of transformer blocks
    "dropout": 0.1,
    "batch_size": 32,
    "max_steps": 2000,      # raise this for a better model, lower for a quick test
    "eval_every": 200,
    "learning_rate": 3e-4,
    "checkpoint_path": "checkpoint.pt",
}


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    return train_data, val_data, vocab_size, encode, decode


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, eval_iters=20):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = os.path.join(os.path.dirname(__file__), cfg["data_file"])
    train_data, val_data, vocab_size, encode, decode = load_data(data_path)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Training on {len(train_data):,} characters")

    model = GPT(
        vocab_size=vocab_size,
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    start = time.time()
    for step in range(cfg["max_steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            elapsed = time.time() - start
            print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed")

        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "vocab": {"stoi": {c: i for i, c in enumerate(sorted(set(open(data_path, encoding='utf-8').read())))}},
    }, cfg["checkpoint_path"])
    print(f"\nSaved checkpoint to {cfg['checkpoint_path']}")

    print("\n--- Sample generation ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start from a single blank token
    generated = model.generate(context, max_new_tokens=400, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
