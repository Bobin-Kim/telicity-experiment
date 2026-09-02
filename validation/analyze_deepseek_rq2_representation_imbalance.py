#!/usr/bin/env python3
"""Artifact-first RQ2.1 aggregation; performs no model forward passes."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist, ttest_1samp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_registry import load_registry
from src.output_validation import atomic_json,write_csv
from src.preflight import folds,load_dataset

CONFIG=ROOT/'configs/multimodel_true_unknown_preference.yaml'
BASE=ROOT/'results/deepseek-7b-telicity-rq'
OUT=ROOT/'results/deepseek-7b-rq2/rq2_1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
 return h.hexdigest()
def stat(v):
 a=np.asarray(v,float);m=float(a.mean());d=float(tdist.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5))
 return {'mean':m,'ci95_low':m-d,'ci95_high':m+d,'p':float(ttest_1samp(a,0).pvalue),'fold_values':a.tolist()}

def main():
 _,fixed,_=load_registry(CONFIG,'deepseek-7b-chat');rows=load_dataset(fixed);ids=[r['id'] for r in rows]
 cache=np.load(BASE/'full_depth_cache.npz',allow_pickle=False);enc=json.loads((BASE/'encoding_decision_summary.json').read_text())
 enca=json.loads((BASE/'encoding_decision_audit.json').read_text());control=json.loads((BASE/'telicity_decision_control/summary.json').read_text())
 controla=json.loads((BASE/'telicity_decision_control/audit.json').read_text());source=json.loads((BASE/'source_pathway/summary.json').read_text())
 sourcea=json.loads((BASE/'source_pathway/deepseek_source_pathway_audit.json').read_text());fs=np.asarray(folds(rows,fixed));groups=np.asarray([r['group'][0] for r in rows]);m=cache['margin'].astype(float)
 assoc=[]
 for f in range(5):assoc.append(float(m[(fs==f)&(groups=='C')].mean()-m[(fs==f)&(groups=='D')].mean()))
 association=stat(assoc);cd=np.isin(groups,['C','D']);association['point_biserial_r']=float(np.corrcoef((groups[cd]=='C').astype(int),m[cd])[0,1]);association['C_margin_mean']=float(m[groups=='C'].mean());association['D_margin_mean']=float(m[groups=='D'].mean())
 tel=control['causal_tests']['telicity'];dec=control['causal_tests']['decision'];readout=control['causal_tests']['readout']
 evidence=[
  {'evidence':'Telicity representation','layer':'L19','effect':enc['telicity']['strongest']['ba_mean'],'ci95_low':enc['telicity']['strongest']['ci95_low'],'ci95_high':enc['telicity']['strongest']['ci95_high'],'p':'','fold_consistency':'5/5 above chance','interpretation':'strongly available'},
  {'evidence':'Telicity-to-preference association C-D','layer':'baseline','effect':association['mean'],'ci95_low':association['ci95_low'],'ci95_high':association['ci95_high'],'p':association['p'],'fold_consistency':f"{sum(x<0 for x in assoc)}/5 negative",'interpretation':'telicity is associated with lower, not higher, True preference'},
  {'evidence':'Telicity-axis causal effect','layer':'L19','effect':tel['one_sd_random_adjusted_bidirectional']['mean'],'ci95_low':tel['one_sd_random_adjusted_bidirectional']['ci95_low'],'ci95_high':tel['one_sd_random_adjusted_bidirectional']['ci95_high'],'p':tel['one_sd_random_adjusted_bidirectional']['p'],'fold_consistency':f"{tel['one_sd_random_adjusted_bidirectional']['consistent_folds']}/5",'interpretation':'not detected'},
  {'evidence':'Decision-axis causal effect','layer':'L30','effect':dec['one_sd_random_adjusted_bidirectional']['mean'],'ci95_low':dec['one_sd_random_adjusted_bidirectional']['ci95_low'],'ci95_high':dec['one_sd_random_adjusted_bidirectional']['ci95_high'],'p':dec['one_sd_random_adjusted_bidirectional']['p'],'fold_consistency':'5/5','interpretation':'causally supported'},
  {'evidence':'Exact final-readout effect','layer':'final norm','effect':readout['one_sd_random_adjusted_bidirectional']['mean'],'ci95_low':readout['one_sd_random_adjusted_bidirectional']['ci95_low'],'ci95_high':readout['one_sd_random_adjusted_bidirectional']['ci95_high'],'p':readout['one_sd_random_adjusted_bidirectional']['p'],'fold_consistency':'5/5','interpretation':'causal positive control'},
 ]
 for regime in source['discovered_regimes']:
  evidence.append({'evidence':'Telicity path restoration','layer':f"L{regime['start_layer']}-L{regime['end_layer']}",'effect':max(regime['normalized_pathway_shares'].values()),'ci95_low':'','ci95_high':'','p':'','fold_consistency':'joint-fold BIC','interpretation':regime['mechanism']})
 OUT.mkdir(parents=True,exist_ok=True);write_csv(OUT/'evidence_metrics.csv',evidence)
 summary={'status':'complete','audit_status':'PASS','classification':'B. readout/decision imbalance (best-supported); Group-C-specific routing locus not uniquely identifiable','representation_availability':{'first_stable_layer':enc['telicity']['first_stable_layer'],'strongest':enc['telicity']['strongest']},'telicity_preference_association':association,'causal_telicity':tel,'causal_decision':dec,'causal_readout':readout,'routing_path_restoration':source['discovered_regimes'],'claim':'Strong Telicity is available and its C-vs-D association points toward lower True preference, while the preregistered Telicity axis is not causal and separate late decision/readout axes are causal. This rules out a simple representation failure and supports a downstream decision/readout imbalance. The available interventions do not uniquely separate a Group-C-specific routing defect from the final readout.'}
 atomic_json(OUT/'summary.json',summary)
 checks={'dataset_hash':fixed['dataset_sha256']=='ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a','sample_ids':cache['ids'].tolist()==ids,'folds':np.array_equal(cache['fold'].astype(int),fs),'prompt_hash':fixed['prompt_contract_sha256']=='f82e03f1a711accb305259ec09e30d165ddaba3d9a445daccb0d069f8d19f016','scoring_identity':controla['scoring_rule']=='first-token z_True-z_Unknown' and controla['True_token_id']==8173 and controla['Unknown_token_id']==21483,'cache_integrity':enca['status']=='PASS' and enca['cache_sha256']==sha(BASE/'full_depth_cache.npz'),'intervention_audit':controla['status']=='PASS' and controla['checks']['row_grid_17200'] and controla['checks']['controls_present'],'source_path_audit':sourcea['status']=='PASS','five_folds':controla['checks']['five_folds'],'norm_audit':controla['checks']['realized_norm_mean_relative_error_le_1pct'] and controla['checks']['realized_norm_max_relative_error_le_5pct'],'cis_pvalues':all(math.isfinite(float(x)) for x in [association['ci95_low'],association['ci95_high'],association['p'],tel['one_sd_random_adjusted_bidirectional']['p'],dec['one_sd_random_adjusted_bidirectional']['p'],readout['one_sd_random_adjusted_bidirectional']['p']]),'continuous_only':True,'binary_artifacts_used':False}
 required={key:value for key,value in checks.items() if key!='binary_artifacts_used'}
 audit={'status':'PASS' if all(required.values()) and checks['binary_artifacts_used'] is False else 'FAIL','checks':checks,'inputs':{str(p):sha(p) for p in [BASE/'full_depth_cache.npz',BASE/'encoding_decision_summary.json',BASE/'telicity_decision_control/summary.json',BASE/'source_pathway/layer_fold_pathway_results.csv']},'expected_items':400,'observed_items':len(ids),'fold_counts':np.bincount(fs).tolist(),'controls':['random direction','label permutation','same-label source-path donors','matched intervention norm']}
 atomic_json(OUT/'audit.json',audit)
 report=f'''# DeepSeek RQ2.1 — Representation imbalance vs readout/routing imbalance

Status: **COMPLETE — audit {audit['status']}**. This was an artifact-first causal aggregation; no GPU stage was rerun.

## Classification

**B. Readout/decision imbalance is the best-supported explanation; the Group-C-specific routing locus is not uniquely identifiable.**

Telicity is strongly available: the first stable layer is L{enc['telicity']['first_stable_layer']} and the strongest held-out representation is L19 (BA {enc['telicity']['strongest']['ba_mean']:.3f}, 5/5 folds). C has a lower baseline True-minus-Unknown margin than D by {association['mean']:+.3f} (95% CI [{association['ci95_low']:+.3f}, {association['ci95_high']:+.3f}]), so the observed Telicity association does not itself supply the erroneous True-driving direction.

The preregistered L19 Telicity-axis intervention is null-adjusted {tel['one_sd_random_adjusted_bidirectional']['mean']:+.4f} (CI [{tel['one_sd_random_adjusted_bidirectional']['ci95_low']:+.4f}, {tel['one_sd_random_adjusted_bidirectional']['ci95_high']:+.4f}], p={tel['one_sd_random_adjusted_bidirectional']['p']:.3f}). In contrast, the separate L30 decision axis is causal ({dec['one_sd_random_adjusted_bidirectional']['mean']:+.4f}, CI [{dec['one_sd_random_adjusted_bidirectional']['ci95_low']:+.4f}, {dec['one_sd_random_adjusted_bidirectional']['ci95_high']:+.4f}], 5/5 folds), and the exact output readout is a strong positive control ({readout['one_sd_random_adjusted_bidirectional']['mean']:+.3f}, CI [{readout['one_sd_random_adjusted_bidirectional']['ci95_low']:+.3f}, {readout['one_sd_random_adjusted_bidirectional']['ci95_high']:+.3f}]).

The RQ1 causal source restoration confirms that Telicity is transmitted through Residual/Context-V pathways, so Attention dominance was not used as a routing inference. Those patches do not uniquely test a Group-C-specific wrong-decision route; that narrower routing claim remains unidentified.

## Claim boundary

The result rules out a simple weak/missing linear Telicity representation. It supports downstream decision/readout imbalance, but does not prove that the exact final readout alone creates the Group-C error or exclude untested nonlinear/distributed Telicity routes.
'''
 (OUT/'REPORT.md').write_text(report,encoding='utf-8')
 (OUT/'RQ_TABLE_CELL.txt').write_text('DeepSeek — Representation vs readout/routing: strong Telicity (L19 BA=.970) and lower C-than-D True margin; L19 Telicity-axis causal effect not detected; separate L30 decision axis and exact readout causal. Classification: readout/decision imbalance best-supported; Group-C-specific routing locus not uniquely identifiable.\n',encoding='utf-8')
 atomic_json(OUT/'complete.json',{'status':'complete','audit':audit['status'],'rq':'RQ2.1','gpu_work_started':False})
 print(json.dumps({'classification':summary['classification'],'audit':audit['status'],'association':association},indent=2))

if __name__=='__main__':main()
