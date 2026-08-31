import json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_registry import load_registry
from src.preflight import load_dataset,folds,build_prompt
from src.preference_scorer import FirstAnswerTokenMargin
from src.output_validation import ensure_clean_or_resume
import yaml

class Tok:
    def __call__(self,text,add_special_tokens=False,**kw):
        class X:pass
        x=X();x.input_ids={"True":[7],"False":[8],"Unknown":[9,10]}.get(text,list(range(max(1,len(text.split())))));return x
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):return "<S>"+messages[0]["content"]+"<U>"+messages[1]["content"]+"<A>"

def main():
    _,fixed,spec=load_registry(ROOT/"configs/multimodel_true_unknown_preference.yaml","qwen2.5-7b")
    rows=load_dataset(fixed);assert len(rows)==400;assert [folds(rows,fixed).count(i) for i in range(5)]==[80]*5
    s=FirstAnswerTokenMargin(Tok());assert s.ids["True"]==7 and s.ids["Unknown"]==9 and s.audit()["alternatives_distinguishable"]
    assert fixed["system_prompt"] in build_prompt(Tok(),fixed,rows[0]["premise"],rows[0]["hypothesis"])
    assert "binary" not in json.dumps({"fixed":fixed,"spec":spec}).lower()
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"bad.yaml";raw=yaml.safe_load((ROOT/"configs/multimodel_true_unknown_preference.yaml").read_text());raw["fixed_experiment_config"]["mode"]="binary_probe";p.write_text(yaml.safe_dump(raw))
        try:load_registry(p,"qwen2.5-7b");raise AssertionError("binary config accepted")
        except ValueError as e:assert "binary" in str(e)
        d=Path(td)/"collision";d.mkdir();(d/"foreign.txt").write_text("x")
        try:ensure_clean_or_resume(d,"abc",False);raise AssertionError("collision accepted")
        except RuntimeError:pass
    class Same(Tok):
        def __call__(self,text,**kw):
            x=super().__call__(text,**kw);x.input_ids=[7];return x
    try:FirstAnswerTokenMargin(Same());raise AssertionError("indistinguishable alternatives accepted")
    except ValueError:pass
    print("PASS core deterministic fixture")
if __name__=="__main__":main()
