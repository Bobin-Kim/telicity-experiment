#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.name=='MANIFEST.json' or '__pycache__' in p.parts or p.suffix=='.pyc':continue
    files.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
canonical_paths=[
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/qwen_full_depth_pipeline.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_qwen_rq2bc_continuous.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_qwen_full_depth_true_signal.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/analyze_qwen_full_depth_stage_a.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_qwen_onset_components.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/analyze_qwen_onset_components.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_qwen_true_signal_mediation.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/aggregate_qwen_full_depth_true_signal.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_upstream_true_origin.py',
  '/home/bobin/imperfective_paradox_repro/run_reproduction.py',
  '/home/bobin/imperfective_paradox_repro/telicity_intervention/run_telicity_subspace_removal.py']
manifest={
 'package':'MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER','package_version':'1.0.0','generation_date':'2026-08-31',
 'dataset_sha256':'ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a',
 'preference_only':True,'scoring_mode':'first_answer_token_logit_margin',
 'canonical_sources':[{'path':x,'sha256':hashlib.sha256(Path(x).read_bytes()).hexdigest()} for x in canonical_paths],
 'validation_status':{'qwen2.5-7b':'PASS source-equivalence + live full-hook smoke','mistral-7b-v03':'PASS protocol + live full-hook smoke','llama3.1-8b':'BLOCKED gated access','yi1.5-9b':'PASS tokenizer/config dry-run','deepseek-7b-chat':'PASS tokenizer/config dry-run','gemma2-9b':'BLOCKED gated access','glm4-9b':'PASS tokenizer/config dry-run'},
 'files':files+ [{'path':'MANIFEST.json','sha256':None,'note':'self-referential manifest; verify all other entries'}]
}
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
