from __future__ import annotations
import torch

class FirstAnswerTokenMargin:
    mode = "first_answer_token_logit_margin"
    def __init__(self, tokenizer, labels=("True", "False", "Unknown")):
        self.labels = tuple(labels)
        self.sequences = {x: tokenizer(x, add_special_tokens=False).input_ids for x in self.labels}
        if any(not ids for ids in self.sequences.values()): raise ValueError("Empty verbalizer tokenization")
        self.ids = {x: ids[0] for x, ids in self.sequences.items()}
        if self.ids["True"] == self.ids["Unknown"]:
            raise ValueError("True and Unknown are indistinguishable under locked first-token scoring")
    def positions(self, attention_mask):
        pos = attention_mask.long().sum(1) - 1
        if (pos < 0).any(): raise ValueError("Invalid answer position")
        return pos
    def from_logits(self, logits, positions):
        ix = torch.arange(logits.shape[0], device=logits.device)
        z = logits[ix, positions]
        t, u = z[:, self.ids["True"]].float(), z[:, self.ids["Unknown"]].float()
        return {"true_logit": t, "unknown_logit": u, "margin": t-u}
    def audit(self):
        return {"mode": self.mode, "formula": "z_True - z_Unknown", "verbalizer_token_ids": self.sequences,
                "scoring_token_ids": {"True": self.ids["True"], "Unknown": self.ids["Unknown"]},
                "alternatives_distinguishable": self.ids["True"] != self.ids["Unknown"]}
