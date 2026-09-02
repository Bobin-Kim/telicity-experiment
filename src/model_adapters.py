from __future__ import annotations
from contextlib import contextmanager

def _get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj

class HFDecoderAdapter:
    """Common interface for Qwen2/Mistral/Llama/Yi/DeepSeek/Gemma2/GLM HF decoders."""
    def __init__(self, model, spec):
        self.model, self.spec = model, spec
        self.root = getattr(model, "model")
        m = spec["modules"]
        self.layers = _get(model, m["layers"])
        self.final_norm = _get(model, m["final_norm"])
        self.output_head = _get(model, m["output_head"])
        g = spec["geometry"]
        if len(self.layers) != g["layers"]:
            raise ValueError(f"Unexpected layer count: configured {g['layers']}, loaded {len(self.layers)}")
        cfg = model.config
        for name, expected in (("hidden_size", g["hidden_size"]), ("num_attention_heads", g["attention_heads"])):
            if int(getattr(cfg, name)) != expected:
                raise ValueError(f"Unexpected {name}: configured {expected}, loaded {getattr(cfg, name)}")

    @property
    def num_layers(self): return len(self.layers)
    @property
    def hidden_size(self): return self.spec["geometry"]["hidden_size"]
    @property
    def num_heads(self): return self.spec["geometry"]["attention_heads"]
    @property
    def head_dim(self): return self.spec["geometry"]["head_dim"]
    def attention(self, layer): return _get(self.layers[layer], self.spec["modules"]["attention"])
    def mlp(self, layer): return _get(self.layers[layer], self.spec["modules"]["mlp"])
    def head_input(self, layer): return _get(self.layers[layer], self.spec["modules"]["head_output_input"])
    def input_norm(self, layer): return _get(self.layers[layer], self.spec["modules"]["input_norm"])
    def query(self, layer): return _get(self.layers[layer], self.spec["modules"]["query"])
    def key(self, layer): return _get(self.layers[layer], self.spec["modules"]["key"])
    def value(self, layer): return _get(self.layers[layer], self.spec["modules"]["value"])
    def gate(self, layer): return _get(self.layers[layer], self.spec["modules"]["gate"])
    def module_inventory(self):
        return {"layers": self.num_layers, "hidden_size": self.hidden_size,
                "attention": [self.attention(i).__class__.__name__ for i in range(self.num_layers)],
                "mlp": [self.mlp(i).__class__.__name__ for i in range(self.num_layers)],
                "source_pathway_modules": {
                    "input_norm": [self.input_norm(i).__class__.__name__ for i in range(self.num_layers)],
                    "query": [self.query(i).__class__.__name__ for i in range(self.num_layers)],
                    "key": [self.key(i).__class__.__name__ for i in range(self.num_layers)],
                    "value": [self.value(i).__class__.__name__ for i in range(self.num_layers)],
                    "gate": [self.gate(i).__class__.__name__ for i in range(self.num_layers)],
                },
                "final_norm": self.final_norm.__class__.__name__, "output_head": self.output_head.__class__.__name__}
    @contextmanager
    def hooks(self, handles):
        try: yield
        finally:
            for handle in handles: handle.remove()

def make_adapter(model, spec):
    if spec.get("adapter") != "hf_decoder": raise ValueError(f"Unresolved adapter {spec.get('adapter')}")
    return HFDecoderAdapter(model, spec)
