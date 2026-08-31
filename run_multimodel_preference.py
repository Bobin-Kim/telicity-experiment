#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from src.model_registry import fingerprint, load_registry, resolved_source
from src.output_validation import atomic_json, ensure_clean_or_resume

def code_fingerprint():
    h=hashlib.sha256()
    for p in sorted([ROOT/"run_multimodel_preference.py",*list((ROOT/"src").glob("*.py"))]):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description="Preference-only True–Unknown multi-model runner")
    p.add_argument("--config",type=Path,required=True);p.add_argument("--model",required=True);p.add_argument("--gpus",default="0")
    p.add_argument("--stages",help="Comma-separated canonical stages; add upstream_source explicitly")
    p.add_argument("--output-dir",type=Path);p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--batch-size",type=int,default=4);p.add_argument("--max-items",type=int,help="Engineering smoke fixture only; outputs are marked non-official")
    a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]=a.gpus
    raw,fixed,spec=load_registry(a.config.resolve(),a.model)
    stages=fixed["stage_order"] if not a.stages else [x.strip() for x in a.stages.split(",") if x.strip()]
    allowed=set(fixed["stage_order"]+fixed.get("optional_stages",[]));bad=set(stages)-allowed
    if bad:raise SystemExit(f"Unknown/out-of-scope stage(s): {sorted(bad)}")
    run_fp=fingerprint({"fixed":fixed,"model_key":a.model,"model":spec,"code":code_fingerprint(),"max_items":a.max_items})
    out=(a.output_dir or ROOT/"results"/a.model).resolve()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    source=resolved_source(spec);common={"revision":spec.get("revision","main"),"trust_remote_code":bool(spec.get("trust_remote_code",False))}
    tokenizer_source=source if spec.get("local_path") and Path(spec["local_path"]).is_dir() else spec.get("tokenizer_id",source)
    try:tok=AutoTokenizer.from_pretrained(tokenizer_source,use_fast=bool(spec.get("tokenizer",{}).get("use_fast",True)),**common)
    except Exception as e:raise SystemExit(f"Tokenizer/config access failed for {a.model} ({source}). Install/cache weights or authenticate with `huggingface-cli login`. Detail: {e}")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side=spec.get("tokenizer",{}).get("padding_side","right")
    from src.preflight import run_preflight
    if a.dry_run:
        report,rows,scorer=run_preflight(tok,fixed,spec,None);report.update({"mode":"dry-run/config-and-tokenizer","model":a.model,"source":source,"run_fingerprint":run_fp,"stages":stages,"official":False})
        print(json.dumps(report,indent=2));return
    import torch
    dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[spec["dtype"]]
    try:model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=dtype,device_map=spec.get("device_map","auto"),low_cpu_mem_usage=True,**common).eval()
    except Exception as e:raise SystemExit(f"Model loading failed for {a.model} ({source}). Check access, disk, RAM/VRAM, and HF authentication. Detail: {e}")
    report,rows,scorer=run_preflight(tok,fixed,spec,model)
    if a.max_items:
        if a.max_items<1:raise SystemExit("--max-items must be positive")
        rows=rows[:a.max_items];report["official"]=False;report["smoke_fixture_items"]=len(rows)
    else:report["official"]=True
    ensure_clean_or_resume(out,run_fp,a.resume);report.update({"model":a.model,"source":source,"run_fingerprint":run_fp,"code_fingerprint":code_fingerprint(),"resolved_model_config":spec})
    atomic_json(out/"preflight.json",report);atomic_json(out/"resolved_config.json",{"package_version":raw["package_version"],"fixed_experiment_config":fixed,"selected_model":a.model,"model_config":spec,"run_fingerprint":run_fp,"code_fingerprint":code_fingerprint()})
    if a.preflight_only:print(f"PASS: preflight written to {out/'preflight.json'}");return
    from src.model_adapters import make_adapter
    from src.pipeline_controller import PipelineController
    ctl=PipelineController(model,tok,make_adapter(model,spec),scorer,rows,fixed,out,run_fp,a.batch_size);ctl.run(stages,a.resume)
    official=not bool(a.max_items)
    if (out/"generator_selection.json").exists():official=official and bool(json.loads((out/"generator_selection.json").read_text()).get("official"))
    atomic_json(out/"RUN_COMPLETE.json",{"status":"complete","official":official,"model":a.model,"stages":stages,"run_fingerprint":run_fp})
    print(f"COMPLETE: {out}")
if __name__=="__main__":main()
