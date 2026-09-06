import torch
import torch.nn.functional as F

from sim import random_episode, ACTIONS
from model import TinyDynamicsModel, state_to_tokens, build_vocab, FIELDS

torch.manual_seed(0)

vocab, stoi = build_vocab(ACTIONS)
V = len(vocab)
NSTATE = len(FIELDS) * 2      # e.g. "room 0 x 3 y 5 hp 3 item 0" -> 10 tokens
SEQLEN = NSTATE + 1 + 1 + NSTATE   # state + action + <bos> + next_state


def encode_example(s, a, s2):
    toks = state_to_tokens(s) + [a] + ["<bos>"] + state_to_tokens(s2)
    return [stoi[t] for t in toks]


def build_dataset(n_episodes=60, ep_len=20):
    data = []
    for ep in range(n_episodes):
        for s, a, s2 in random_episode(length=ep_len, seed=ep):
            data.append(encode_example(s, a, s2))
    return torch.tensor(data, dtype=torch.long)


def main():
    data = build_dataset()
    print(f"vocab size: {V}, examples: {len(data)}, seq len: {SEQLEN}")

    model = TinyDynamicsModel(vocab_size=V, max_len=SEQLEN, d_model=32, n_head=2, n_layer=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # standard next-token LM loss over the whole sequence: predicting the
    # state tokens teaches next-token structure, predicting the action is a
    # trivial freebie, predicting the *next state* tokens (positions
    # NSTATE+2 .. end) is the actual dynamics-learning objective.
    n_epochs = 10
    batch_size = 128
    for epoch in range(n_epochs):
        perm = data[torch.randperm(len(data))]
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            batch = perm[i:i + batch_size]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}")

    torch.save(
        {"model_state": model.state_dict(), "vocab": vocab, "stoi": stoi},
        "checkpoint.pt",
    )
    print("saved checkpoint.pt")


if __name__ == "__main__":
    main()
