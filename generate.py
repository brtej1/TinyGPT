"""
Generate text from a trained checkpoint (run train.py first to create one).

Run with:  python generate.py
Or:        python generate.py "ROMEO:" 300     (prompt, number of characters)
"""

import sys
import torch
from model import GPT


def main():
    checkpoint_path = "checkpoint.pt"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stoi = ckpt["vocab"]["stoi"]
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)

    model = GPT(
        vocab_size=len(stoi),
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=0.0,  # no dropout at inference time
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    context = torch.tensor([encode(prompt)], dtype=torch.long)
    out = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
