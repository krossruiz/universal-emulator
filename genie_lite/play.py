"""
REPL "emulator": type actions, the trained dynamics model predicts the next
state autoregressively, no pixels involved anywhere. Compare its guesses
against sim.step() (the real ground truth) with `diff` to see where the
learned model's imagination diverges from the real logic.
"""
import torch

from sim import initial_state, step, ACTIONS
from model import TinyDynamicsModel, state_to_tokens, FIELDS

ckpt = torch.load("checkpoint.pt", map_location="cpu")
vocab, stoi = ckpt["vocab"], ckpt["stoi"]
itos = {i: t for t, i in stoi.items()}
V = len(vocab)
NSTATE = len(FIELDS) * 2
SEQLEN = NSTATE + 1 + 1 + NSTATE

model = TinyDynamicsModel(vocab_size=V, max_len=SEQLEN, d_model=32, n_head=2, n_layer=2)
model.load_state_dict(ckpt["model_state"])
model.eval()


def tokens_to_state(toks):
    d = {}
    for i in range(0, len(toks), 2):
        d[toks[i]] = int(toks[i + 1])
    return d


@torch.no_grad()
def predict_next(state, action):
    ids = [stoi[t] for t in state_to_tokens(state) + [action] + ["<bos>"]]
    ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(NSTATE):
        logits = model(ids)
        next_id = torch.argmax(logits[0, -1]).item()
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    out_toks = [itos[i] for i in ids[0, -NSTATE:].tolist()]
    return tokens_to_state(out_toks)


def main():
    real = initial_state()
    imagined = dict(real)
    print("genie_lite REPL - actions:", ", ".join(ACTIONS), " (or 'quit')")
    print("real     :", real)
    print("imagined :", imagined)
    while True:
        a = input("> ").strip().lower()
        if a in ("quit", "exit", "q"):
            break
        if a not in ACTIONS:
            print("unknown action, choices:", ACTIONS)
            continue
        real = step(real, a)
        imagined = predict_next(imagined, a)
        match = "OK " if imagined == real else "DIFF"
        print(f"[{match}] real     : {real}")
        print(f"[{match}] imagined : {imagined}")


if __name__ == "__main__":
    main()
