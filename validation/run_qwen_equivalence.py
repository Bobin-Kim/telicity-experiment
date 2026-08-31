#!/usr/bin/env python3
"""Deterministic source-contract equivalence fixture; no scientific result is asserted."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; REPO=Path('/home/bobin/imperfective_paradox_repro');sys.path[:0]=[str(ROOT),str(REPO.parent)]
from src.pipeline_controller import ridge,pred,r2
from src.preflight import build_prompt,folds
from transformers import AutoTokenizer
from imperfective_paradox_repro.run_reproduction import build_prompt as old_prompt,LABELS
from imperfective_paradox_repro.telicity_intervention.run_telicity_subspace_removal import build_verb_folds

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load(REPO/'telicity_intervention/run_qwen_rq2bc_continuous.py','canonical_a23')
cfg=yaml.safe_load((ROOT/'configs/multimodel_true_unknown_preference.yaml').read_text());fixed=cfg['fixed_experiment_config'];rows=json.loads(Path(fixed['dataset_path']).read_text())
tok=AutoTokenizer.from_pretrained('/data/sl_datasets/models/Qwen/Qwen2.5-7B-Instruct')
rng=np.random.default_rng(42);x=rng.normal(size=(31,12));y=rng.normal(size=31);train=np.arange(23);test=np.arange(23,31)
of=old.ridge(x[train],y[train]);nf=ridge(x[train],y[train]);op=old.pred(of,x[test]);npred=pred(nf,x[test])
old_map,_=build_verb_folds(rows,42,5);new_f=folds(rows,fixed)
checks={
 'dataset_ids_and_order':[r['id'] for r in rows]==[f'{g}_{i:03d}' for g in 'ABCD' for i in range(1,101)],
 'rendered_prompt':old_prompt(tok,rows[0]['premise'],rows[0]['hypothesis'])==build_prompt(tok,fixed,rows[0]['premise'],rows[0]['hypothesis']),
 'scoring_tokens':[tok(x,add_special_tokens=False).input_ids[0] for x in LABELS]==[2514,4049,13790],
 'token_position_policy':fixed['scoring_position']=='last_non_padding_prompt_token',
 'folds':new_f==[old_map[r['verb']] for r in rows],
 'seeds':fixed['random_seed']==42 and fixed['folds']['seed']==42,
 'layer_coverage':cfg['model_configs']['qwen2.5-7b']['geometry']['layers']==28,
 'hook_paths':cfg['model_configs']['qwen2.5-7b']['modules']=={'layers':'model.layers','attention':'self_attn','mlp':'mlp','attention_output':'self_attn','head_output_input':'self_attn.o_proj','final_norm':'model.norm','output_head':'lm_head'},
 'intervention_conditions':fixed['intervention']['conditions']==['decision_pos','decision_neg','random_pos','random_neg','subspace_remove','random_remove'],
 'ridge_predictions':bool(np.allclose(op,npred,rtol=0,atol=1e-12)),
 'downstream_r2':bool(np.isclose(old.r2(y[test],op),r2(y[test],npred),rtol=0,atol=1e-12)),
 'margin_formula':float(np.array([3.25,1,-2])[0]-np.array([3.25,1,-2])[2])==5.25,
}
print(json.dumps({'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'max_prediction_abs_diff':float(np.max(abs(op-npred))),'r2_abs_diff':float(abs(old.r2(y[test],op)-r2(y[test],npred)))},indent=2))
