# TinyGPT — a complete LLM, just a small one

This is a real GPT-style language model: same architecture as production
LLMs, scaled down so it trains on a laptop CPU in a few minutes instead of
needing a data center. It learns to generate Shakespeare-ish text,
character by character.

## How the files map to "what a complete LLM looks like"

| Piece                         | File                         |
|--------------------------------|-------------------------------|
| Tokenizer                     | built into `train.py` (character-level — each unique character is one token) |
| Embedding layer, transformer blocks, output head | `model.py` |
| Training loop (data → loss → backprop → updated weights) | `train.py` |
| The trained model itself (architecture config + weights + vocab) | `checkpoint.pt`, created when you run `train.py` |
| Generation / inference        | `generate.py`                |

## Setup (one-time)

```bash
pip install torch
```

That's the only dependency.

## Train it

```bash
python train.py
```

With the default settings (4 layers, 128-dim embeddings, 2000 steps) this
takes roughly 5–15 minutes on a modern laptop CPU. You'll see the loss
drop every 200 steps, and at the end it prints a sample of generated text
and saves `checkpoint.pt`.

To experiment faster while you're learning, open `train.py` and lower
`max_steps` to something like `300` — you'll see it run in under a minute,
though the output will be closer to gibberish since it hasn't trained long
enough to learn structure.

## Generate more text later (without retraining)

```bash
python generate.py
python generate.py "ROMEO:" 500
```

The first argument is a prompt to continue from, the second is how many
characters to generate.

## What to tweak to learn more

- `n_layer` / `n_head` / `n_embd` in `CONFIG` — make the model bigger or
  smaller and watch how it affects both training speed and the quality of
  generated text.
- `block_size` — how much context (how many previous characters) the model
  can see at once.
- Swap in your own text file as `data_file` — train it on a book, your own
  writing, song lyrics you have rights to, anything — and watch the
  generated text take on that style.
- This model is character-level for simplicity. Real LLMs use *subword*
  tokenizers (like Byte-Pair Encoding) so each token is closer to a word
  fragment — that's the next thing worth exploring once this clicks.
