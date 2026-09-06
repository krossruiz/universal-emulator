"""
Tiny Genie-style dynamics model, minus everything visual.

Real Genie: video tokenizer (frames <-> latents) + latent action model +
autoregressive transformer that predicts the next latent frame token given
past latents and the (inferred) action.

Here: no video tokenizer at all. State is a short sequence of text tokens
("room 0 x 3 y 5 hp 3 item 0"). The transformer's only job is the dynamics
part Genie is actually famous for: autoregressively predicting the next
state's tokens given the current state's tokens + an action token. Same
"autoregressive, action-conditioned next-step predictor" shape, at a scale
that trains on a CPU in seconds.
"""
import torch
import torch.nn as nn

FIELDS = ["room", "x", "y", "hp", "item"]


def state_to_tokens(s):
    toks = []
    for f in FIELDS:
        toks.append(f)
        toks.append(str(s[f]))
    return toks


def build_vocab(actions):
    vocab = ["<bos>", "<pad>"]
    vocab += FIELDS
    vocab += [str(i) for i in range(8)]   # covers room/x/y/hp/item ranges used
    vocab += list(actions)
    vocab = sorted(set(vocab))
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


class TinyDynamicsModel(nn.Module):
    """Decoder-only transformer, ~ a few hundred k params."""

    def __init__(self, vocab_size, d_model=64, n_head=4, n_layer=3, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.register_buffer(
            "_mask", nn.Transformer.generate_square_subsequent_mask(max_len), persistent=False
        )

    def forward(self, tok_ids):
        b, t = tok_ids.shape
        pos = torch.arange(t, device=tok_ids.device).unsqueeze(0)
        x = self.tok_emb(tok_ids) + self.pos_emb(pos)
        mask = self._mask[:t, :t]
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(x)
