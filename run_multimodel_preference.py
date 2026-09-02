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
    p.add_argument("--stages",help="Comma-separated stages; supplemental upstream/re-emergence stages must be requested explicitly")
    p.add_argument("--output-dir",type=Path);p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--batch-size",type=int,help="Override all stage-specific model batch sizes");p.add_argument("--max-items",type=int,help="Engineering smoke fixture only; outputs are marked non-official")
    p.add_argument("--layers",type=int,nargs="+",help="Engineering layer fixture; normal stages use zero-based block indices, re-emergence stages use documented one-based removal layers")
    p.add_argument("--folds",type=int,nargs="+",help="Engineering intervention fixture only; default is every fold")
    a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]=a.gpus
    raw,fixed,spec=load_registry(a.config.resolve(),a.model)
    stages=fixed["stage_order"] if not a.stages else [x.strip() for x in a.stages.split(",") if x.strip()]
    allowed=set(fixed["stage_order"]+fixed.get("optional_stages",[]));bad=set(stages)-allowed
    if bad:raise SystemExit(f"Unknown/out-of-scope stage(s): {sorted(bad)}")
    run_fp=fingerprint({"fixed":fixed,"model_key":a.model,"model":spec,"code":code_fingerprint(),"max_items":a.max_items,"layers":a.layers,"folds":a.folds})
    out=(a.output_dir or ROOT/"results"/a.model).resolve()
    # Match the canonical scripts: establish deterministic PyTorch/CUDA state
    # before importing Transformers (which imports torch internally).
    import torch,random,numpy as np
    seed=int(fixed["random_seed"]);os.environ.setdefault("PYTHONHASHSEED",str(seed));os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8");random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    try:torch.use_deterministic_algorithms(True)
    except Exception:pass
    from transformers import AutoModelForCausalLM,AutoTokenizer
    source=resolved_source(spec);common={"revision":spec.get("revision","main"),"trust_remote_code":bool(spec.get("trust_remote_code",False))}
    tokenizer_source=source if spec.get("local_path") and Path(spec["local_path"]).is_dir() else spec.get("tokenizer_id",source)
    original_mistral=spec.get("protocol_compatibility",{}).get("load_strategy")=="original_single_device_to_cuda"
    try:
        if original_mistral:tok=AutoTokenizer.from_pretrained(tokenizer_source,local_files_only=True,use_fast=True)
        else:tok=AutoTokenizer.from_pretrained(tokenizer_source,use_fast=bool(spec.get("tokenizer",{}).get("use_fast",True)),**common)
    except Exception as e:raise SystemExit(f"Tokenizer/config access failed for {a.model} ({source}). Install/cache weights or authenticate with `huggingface-cli login`. Detail: {e}")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side=spec.get("tokenizer",{}).get("padding_side","right")
    from src.preflight import run_preflight
    if a.dry_run:
        report,rows,scorer=run_preflight(tok,fixed,spec,None);report.update({"mode":"dry-run/config-and-tokenizer","model":a.model,"source":source,"run_fingerprint":run_fp,"stages":stages,"official":False})
        print(json.dumps(report,indent=2));return
    dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[spec["dtype"]]
    if spec.get("allow_legacy_torch_load_for_trusted_local_checkpoint"):
        if not (spec.get("local_path") and Path(spec["local_path"]).is_dir() and source == spec["local_path"]):
            raise SystemExit("Legacy torch-load compatibility is allowed only for the configured trusted local checkpoint")
        # Transformers 4.52 blocks legacy .bin loading with torch<2.6.  This
        # opt-in is scoped to the pinned local DeepSeek artifact and process.
        import transformers.modeling_utils as _modeling_utils
        _modeling_utils.check_torch_load_is_safe=lambda:None
    try:
        if original_mistral:
            model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=dtype,local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        else:model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=dtype,device_map=spec.get("device_map","auto"),low_cpu_mem_usage=True,**common).eval()
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
    ctl=PipelineController(model,tok,make_adapter(model,spec),scorer,rows,fixed,out,run_fp,a.batch_size,a.layers,a.folds);ctl.run(stages,a.resume)
    official=not bool(a.max_items or a.layers or a.folds)
    if (out/"generator_selection.json").exists():official=official and bool(json.loads((out/"generator_selection.json").read_text()).get("official"))
    atomic_json(out/"RUN_COMPLETE.json",{"status":"complete","official":official,"model":a.model,"stages":stages,"run_fingerprint":run_fp})
    print(f"COMPLETE: {out}")
if __name__=="__main__":main()
