from __future__ import annotations
import hashlib, json, random
from collections import Counter
from pathlib import Path
from .model_adapters import make_adapter
from .preference_scorer import FirstAnswerTokenMargin

def build_prompt(tokenizer, fixed, premise, hypothesis):
    messages = [{"role":"system", "content":fixed["system_prompt"]},
                {"role":"user", "content":fixed["user_format"].format(premise=premise, hypothesis=hypothesis)}]
    try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception: return fixed["system_prompt"] + "\n\n" + messages[1]["content"]

def load_dataset(fixed):
    prompt_hash=hashlib.sha256((fixed["system_prompt"]+"\0"+fixed["user_format"]).encode()).hexdigest()
    if prompt_hash != fixed.get("prompt_contract_sha256") or prompt_hash != "f82e03f1a711accb305259ec09e30d165ddaba3d9a445daccb0d069f8d19f016": raise ValueError("Prompt mismatch against canonical source contract")
    p = Path(fixed["dataset_path"]); digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != fixed["dataset_sha256"]: raise ValueError(f"Dataset hash mismatch: {digest}")
    rows = json.loads(p.read_text())
    ids = [r["id"] for r in rows]
    if len(rows) != fixed["expected_items"] or len(ids) != len(set(ids)): raise ValueError("Missing or duplicated dataset IDs")
    counts = Counter(r["group"][0] for r in rows)
    if dict(counts) != fixed["expected_groups"]: raise ValueError(f"Condition counts mismatch: {counts}")
    expected = [f"{g}_{i:03d}" for g in "ABCD" for i in range(1, 101)]
    if ids != expected: raise ValueError("Dataset item IDs/order differ from canonical A/B/C/D ordering")
    return rows

def folds(rows, fixed):
    n, seed = fixed["folds"]["count"], fixed["folds"]["seed"]
    rng = random.Random(seed); cls = {r["verb"]:(1 if r["group"][0] in "AC" else 0) for r in rows}
    a=[v for v,c in cls.items() if c]; b=[v for v,c in cls.items() if not c]; rng.shuffle(a); rng.shuffle(b)
    by={v:i%n for i,v in enumerate(a)}; by.update({v:i%n for i,v in enumerate(b)})
    return [by[r["verb"]] for r in rows]

def run_preflight(tokenizer, fixed, spec, model=None):
    rows = load_dataset(fixed); scorer = FirstAnswerTokenMargin(tokenizer, fixed["labels"])
    prompt = build_prompt(tokenizer, fixed, rows[0]["premise"], rows[0]["hypothesis"])
    enc = tokenizer(prompt, add_special_tokens=False)
    if not enc.input_ids: raise ValueError("Rendered prompt has no tokens")
    report = {"status":"PASS", "dataset":{"path":fixed["dataset_path"],"sha256":fixed["dataset_sha256"],"items":len(rows),"first_id":rows[0]["id"],"last_id":rows[-1]["id"]},
              "rendered_prompt":prompt, "prompt_token_count":len(enc.input_ids), "scoring_position":len(enc.input_ids)-1,
              "tokenization":scorer.audit(), "fold_counts":dict(Counter(folds(rows,fixed)))}
    if model is not None:
        adapter=make_adapter(model,spec); report["hooks"]=adapter.module_inventory()
    return report, rows, scorer
