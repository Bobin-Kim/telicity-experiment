from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from .preflight import build_prompt, folds
from .output_validation import atomic_json, write_csv, validate_stage

DEPENDENCIES={"behavior":[],"cache":[],"predictive_localization":["cache"],"causal_intervention":["cache","predictive_localization"],"component_analysis":["cache"],"generator_localization":["component_analysis","causal_intervention"],"aggregate":["behavior","predictive_localization","causal_intervention","generator_localization"],"upstream_source":["cache","generator_localization"]}

def ridge(x,y,a=10.):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1; z=(x-mu)/sd; yc=y-y.mean()
    c=np.linalg.solve(z.T@z+a*np.eye(z.shape[1]),z.T@yc) if z.shape[1]<=len(z) else z.T@np.linalg.solve(z@z.T+a*np.eye(len(z)),yc)
    return mu,sd,c,float(y.mean())
def pred(f,x): mu,sd,c,b=f; return ((x-mu)/sd)@c+b
def r2(y,p): return 1-float(((y-p)**2).sum())/max(float(((y-y.mean())**2).sum()),1e-12)
def norm(x): return x/max(float(np.linalg.norm(x)),1e-12)

class PipelineController:
    def __init__(self, model, tokenizer, adapter, scorer, rows, fixed, out, fingerprint, batch_size=4):
        self.model,self.tok,self.a,self.s,self.rows,self.f,self.out,self.fp=model,tokenizer,adapter,scorer,rows,fixed,Path(out),fingerprint
        self.batch=batch_size; self.fold=np.asarray(folds(rows,fixed)); self.device=next(model.parameters()).device
    def marker(self,stage,extra=None): atomic_json(self.out/f"{stage}.complete.json",{"status":"complete","stage":stage,"run_fingerprint":self.fp,**(extra or {})})
    def encode(self, indices):
        p=[build_prompt(self.tok,self.f,self.rows[i]["premise"],self.rows[i]["hypothesis"]) for i in indices]
        e=self.tok(p,return_tensors="pt",padding=True,add_special_tokens=False)
        return {k:v.to(self.device) for k,v in e.items()}
    def forward(self, indices):
        e=self.encode(indices); pos=self.s.positions(e["attention_mask"])
        with torch.inference_mode(): o=self.model(**e,use_cache=False,return_dict=True)
        return self.s.from_logits(o.logits,pos),pos,e
    def behavior(self):
        rows=[]
        for st in range(0,len(self.rows),self.batch):
            ids=list(range(st,min(st+self.batch,len(self.rows)))); z,_,_=self.forward(ids)
            for j,i in enumerate(ids): rows.append({"example_index":i,"example_id":self.rows[i]["id"],"group":self.rows[i]["group"][0],"fold":int(self.fold[i]),"true_logit":float(z["true_logit"][j]),"unknown_logit":float(z["unknown_logit"][j]),"margin":float(z["margin"][j])})
        write_csv(self.out/"behavior.csv",rows); self.marker("behavior",{"rows":len(rows),"scoring_mode":self.s.mode})
    def cache(self):
        R,A,M,H,margin=[],[],[],[],[]; L=self.a.num_layers
        for st in range(0,len(self.rows),self.batch):
            inds=list(range(st,min(st+self.batch,len(self.rows)))); e=self.encode(inds); pos=self.s.positions(e["attention_mask"]); cap={"r":[None]*L,"a":[None]*L,"m":[None]*L,"h":[None]*L}; handles=[]
            def pre(k):
                def fn(_m,inp): cap["r"][k]=inp[0].detach()
                return fn
            def post(kind,k):
                def fn(_m,_i,out): cap[kind][k]=(out[0] if isinstance(out,tuple) else out).detach()
                return fn
            def hpre(k):
                def fn(_m,inp): cap["h"][k]=inp[0].detach()
                return fn
            for l in range(L):
                handles += [self.a.layers[l].register_forward_pre_hook(pre(l)),self.a.attention(l).register_forward_hook(post("a",l)),self.a.mlp(l).register_forward_hook(post("m",l)),self.a.head_input(l).register_forward_pre_hook(hpre(l))]
            try:
                with torch.inference_mode(): o=self.model(**e,use_cache=False,return_dict=True)
            finally:
                for h in handles:h.remove()
            ix=torch.arange(len(inds),device=self.device)
            R.append(torch.stack([cap["r"][l][ix,pos] for l in range(L)],1).half().cpu().numpy())
            A.append(torch.stack([cap["a"][l][ix,pos] for l in range(L)],1).half().cpu().numpy())
            M.append(torch.stack([cap["m"][l][ix,pos] for l in range(L)],1).half().cpu().numpy())
            H.append(torch.stack([cap["h"][l][ix,pos] for l in range(L)],1).half().cpu().numpy())
            margin.append(self.s.from_logits(o.logits,pos)["margin"].cpu().numpy())
        arrays={"residual":np.concatenate(R),"attention":np.concatenate(A),"mlp":np.concatenate(M),"head":np.concatenate(H),"margin":np.concatenate(margin),"fold":self.fold}
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError("NaN/Inf in component cache")
        np.savez_compressed(self.out/"preference_cache.npz",**arrays)
        self.marker("cache",{"rows":len(self.rows),"layers":L,"layer_coverage":list(range(L)),"position_policy":self.f["scoring_position"]})
    def predictive_localization(self):
        z=np.load(self.out/"preference_cache.npz"); x=z["residual"].astype(float); y=z["margin"].astype(float); records=[]; selected=[]
        for held in range(self.f["folds"]["count"]):
            tr=self.fold!=held; te=~tr; scores=[]
            for l in range(self.a.num_layers):
                p=pred(ridge(x[tr,l],y[tr],self.f["preference_basis"]["ridge_alpha"]),x[te,l]); score=r2(y[te],p); records.append({"fold":held,"layer":l,"heldout_r2":score}); inner=[]
                for v in range(self.f["folds"]["count"]):
                    it=tr&(self.fold!=v); va=tr&(self.fold==v)
                    if va.any():inner.append(r2(y[va],pred(ridge(x[it,l],y[it]),x[va,l])))
                scores.append((float(np.mean(inner)),l))
            best=max(scores); selected.append({"heldout_fold":held,"layer":best[1],"inner_cv_r2":best[0],"selection":"all-layer inner-CV; no prior layer"})
        write_csv(self.out/"predictive_full_depth.csv",records); write_csv(self.out/"selected_layers.csv",selected); self.marker("predictive_localization",{"layers":self.a.num_layers,"folds":5})
    def _axis(self,x,y,tr):
        center=x[tr].mean(0); work=x.copy(); basis=[]
        for _ in range(self.f["preference_basis"]["components"]):
            mu,sd,c,_=ridge(work[tr],y[tr]); v=c/sd
            for q in basis:v-=q*np.dot(q,v)
            if np.linalg.norm(v)<1e-10:break
            v=norm(v);basis.append(v);work-=np.outer(work@v,v)
        b=np.stack(basis); axis=norm(b.T@np.linalg.lstsq((x[tr]-center)@b.T,y[tr]-y[tr].mean(),rcond=1e-8)[0]); return center,b,axis,float(np.std((x[tr]-center)@axis) or 1)
    def causal_intervention(self):
        z=np.load(self.out/"preference_cache.npz"); x=z["residual"].astype(float); y=z["margin"].astype(float); rows=[]; seed=self.f["random_seed"]
        for held in range(5):
            tr=self.fold!=held; test=np.flatnonzero(~tr).tolist()
            for layer in range(self.a.num_layers):
                center,basis,axis,sd=self._axis(x[:,layer],y,tr); rng=np.random.default_rng(seed+held*1000+layer); raw=rng.normal(size=(x.shape[-1],len(basis)));raw-=basis.T@(basis@raw);rbase=np.linalg.qr(raw)[0].T;rv=rbase[0];decision_dist=np.linalg.norm(((x[tr,layer]-center)@basis.T)@basis,axis=1).mean();random_dist=np.linalg.norm(((x[tr,layer]-center)@rbase.T)@rbase,axis=1).mean();rscale=float(decision_dist/max(random_dist,1e-12))
                for cond in self.f["intervention"]["conditions"]:
                    for st in range(0,len(test),self.batch):
                        inds=test[st:st+self.batch]; e=self.encode(inds); pos=self.s.positions(e["attention_mask"])
                        def hook(_m,inp):
                            h=inp[0].clone(); ix=torch.arange(len(inds),device=h.device); v=h[ix,pos].float(); ct=torch.tensor(center,device=h.device)
                            if cond=="subspace_remove": v-=((v-ct)@torch.tensor(basis.T,device=h.device))@torch.tensor(basis,device=h.device)
                            elif cond=="random_remove": v-=rscale*((v-ct)@torch.tensor(rbase.T,device=h.device))@torch.tensor(rbase,device=h.device)
                            else:
                                direction=rv if cond.startswith("random") else axis; sign=1 if cond.endswith("pos") else -1; v+=sign*sd*torch.tensor(direction,device=h.device)
                            h[ix,pos]=v.to(h.dtype); return (h,*inp[1:])
                        handle=self.a.layers[layer].register_forward_pre_hook(hook)
                        try:
                            with torch.inference_mode():o=self.model(**e,use_cache=False,return_dict=True)
                        finally:handle.remove()
                        m=self.s.from_logits(o.logits,pos)["margin"].cpu().numpy()
                        for j,i in enumerate(inds):rows.append({"example_index":i,"example_id":self.rows[i]["id"],"group":self.rows[i]["group"][0],"heldout_fold":held,"layer":layer,"condition":cond,"baseline_margin":float(y[i]),"margin":float(m[j]),"delta_margin":float(m[j]-y[i])})
        write_csv(self.out/"causal_full_depth.csv",rows)
        for held in range(5):
            for layer in range(self.a.num_layers):
                shard=[r for r in rows if r["heldout_fold"]==held and r["layer"]==layer];d=self.out/"stage_a"/f"L{layer:02d}"/f"fold{held}";write_csv(d/"interventions.csv",shard);atomic_json(d/"complete.json",{"status":"complete","fold":held,"layer":layer,"rows":len(shard),"run_fingerprint":self.fp})
        self.marker("causal_intervention",{"rows":len(rows),"layers":self.a.num_layers,"conditions":self.f["intervention"]["conditions"]})
    def component_analysis(self):
        z=np.load(self.out/"preference_cache.npz"); y=z["margin"].astype(float); out=[]; heads=[]
        for comp in ("residual","attention","mlp"):
            x=z[comp].astype(float)
            for held in range(5):
                tr=self.fold!=held;te=~tr
                for l in range(self.a.num_layers):
                    inner=[]
                    for v in range(5):
                        it=tr&(self.fold!=v);va=tr&(self.fold==v)
                        if va.any():inner.append(r2(y[va],pred(ridge(x[it,l],y[it]),x[va,l])))
                    out.append({"component":comp,"fold":held,"layer":l,"heldout_r2":r2(y[te],pred(ridge(x[tr,l],y[tr]),x[te,l])),"inner_cv_r2":float(np.mean(inner))})
        hx=z["head"].astype(float).reshape(len(y),self.a.num_layers,self.a.num_heads,self.a.head_dim)
        for held in range(5):
            tr=self.fold!=held;te=~tr
            for l in range(self.a.num_layers):
                for h in range(self.a.num_heads):heads.append({"fold":held,"layer":l,"head":h,"heldout_r2":r2(y[te],pred(ridge(hx[tr,l,h],y[tr]),hx[te,l,h]))})
        write_csv(self.out/"component_full_depth.csv",out);write_csv(self.out/"head_full_depth.csv",heads)
        # Candidate layers are selected independently inside each outer fold, then tested by actual forward intervention.
        causal=[]
        for held in range(5):
            tr=self.fold!=held;test=np.flatnonzero(~tr).tolist()
            for comp in ("attention","mlp"):
                candidates=[r for r in out if r["component"]==comp and r["fold"]==held];layer=max(candidates,key=lambda q:q["inner_cv_r2"])["layer"]
                x=z[comp][:,layer].astype(float);center=x[tr].mean(0);axis=norm((x[tr]-center).T@(y[tr]-y[tr].mean()));sd=float(np.std((x[tr]-center)@axis) or 1)
                rng=np.random.default_rng(self.f["random_seed"]+held*100+layer);rv=rng.normal(size=x.shape[-1]);rv-=axis*np.dot(axis,rv);rv=norm(rv)
                module=self.a.attention(layer) if comp=="attention" else self.a.mlp(layer)
                for cond in ("decision_pos","decision_neg","random_pos","random_neg"):
                    for st in range(0,len(test),self.batch):
                        inds=test[st:st+self.batch];e=self.encode(inds);pos=self.s.positions(e["attention_mask"])
                        def hook(_m,_inp,output):
                            base=output[0] if isinstance(output,tuple) else output;v=base.clone();ix=torch.arange(len(inds),device=v.device);direction=rv if cond.startswith("random") else axis;sign=1 if cond.endswith("pos") else -1;v[ix,pos]+=sign*sd*torch.tensor(direction,device=v.device,dtype=v.dtype);return (v,*output[1:]) if isinstance(output,tuple) else v
                        handle=module.register_forward_hook(hook)
                        try:
                            with torch.inference_mode():o=self.model(**e,use_cache=False,return_dict=True)
                        finally:handle.remove()
                        m=self.s.from_logits(o.logits,pos)["margin"].cpu().numpy()
                        for j,i in enumerate(inds):causal.append({"component":comp,"layer":layer,"head":-1,"heldout_fold":held,"example_id":self.rows[i]["id"],"condition":cond,"baseline_margin":float(y[i]),"margin":float(m[j]),"delta_margin":float(m[j]-y[i])})
        write_csv(self.out/"component_causal.csv",causal)
        for held in range(5):
            shard=[r for r in causal if r["heldout_fold"]==held];d=self.out/"stage_b_onset"/f"fold{held}";write_csv(d/"interventions.csv",shard);entities={(r["component"],r["layer"],r["head"]) for r in shard};atomic_json(d/"complete.json",{"status":"complete","fold":held,"rows":len(shard),"entities":len(entities),"run_fingerprint":self.fp})
        self.marker("component_analysis",{"components":["residual","attention","mlp","heads"],"layers":self.a.num_layers,"causal_controls":["decision_pos","decision_neg","random_pos","random_neg"]})
    def generator_localization(self):
        import csv
        from scipy.stats import t as tdist, ttest_1samp
        with (self.out/"component_causal.csv").open() as f: rows=list(csv.DictReader(f))
        summary=[]
        for comp in ("attention","mlp"):
            fold_effect=[];layers=[]
            for held in range(5):
                z=[r for r in rows if r["component"]==comp and int(r["heldout_fold"])==held];layers.append(int(z[0]["layer"]));pos=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="decision_pos"]);neg=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="decision_neg"]);rp=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="random_pos"]);rn=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="random_neg"]);fold_effect.append((pos-neg)-(rp-rn))
            mean=float(np.mean(fold_effect));se=float(np.std(fold_effect,ddof=1)/np.sqrt(5));ci=float(tdist.ppf(.975,4)*se);p=float(ttest_1samp(fold_effect,0).pvalue)
            summary.append({"component":comp,"median_selected_layer":int(np.median(layers)),"mean_control_adjusted_contrast":mean,"ci95_low":mean-ci,"ci95_high":mean+ci,"two_sided_fold_t_p":p,"positive_folds":sum(x>0 for x in fold_effect)})
        reliable=[r for r in summary if r["positive_folds"]>=4 and r["ci95_low"]>0]; chosen=min(reliable,key=lambda r:(r["median_selected_layer"],0 if r["component"]=="attention" else 1)) if reliable else None
        write_csv(self.out/"generator_candidates.csv",summary); selection={"status":"complete" if chosen else "audit_failure","official":bool(chosen),"earliest_reliable_component":chosen,"criterion":"actual forward decision contrast minus matched random contrast; 95% fold-t CI > 0 and >=4/5 positive folds; no prior component"};atomic_json(self.out/"generator_selection.json",selection);atomic_json(self.out/"stage_b_selection.json",selection);self.marker("generator_localization",{"official":bool(chosen)})
    def upstream_source(self):
        selection=json.loads((self.out/"generator_selection.json").read_text())["earliest_reliable_component"]
        if not selection: raise RuntimeError("Generator audit failed; upstream source is non-interpretable")
        z=np.load(self.out/"preference_cache.npz"); y=z["margin"].astype(float); rows=[]
        for comp in ("attention","mlp"):
            for l in range(int(selection["median_selected_layer"])):
                x=z[comp][:,l].astype(float); rows.append({"component":comp,"layer":l,"correlation":float(np.corrcoef(x@norm((x-x.mean(0)).T@(y-y.mean())),y)[0,1])})
        write_csv(self.out/"upstream_source.csv",rows);self.marker("upstream_source",{"supplemental":True})
    def aggregate(self):
        import csv
        from scipy.stats import t as tdist, ttest_1samp
        with (self.out/"causal_full_depth.csv").open() as f: r=list(csv.DictReader(f)); summary=[]
        for c in self.f["intervention"]["conditions"]:
            for l in range(self.a.num_layers):
                v=[float(x["delta_margin"]) for x in r if x["condition"]==c and int(x["layer"])==l];fv=[np.mean([float(x["delta_margin"]) for x in r if x["condition"]==c and int(x["layer"])==l and int(x["heldout_fold"])==f]) for f in range(5)];mean=float(np.mean(fv));se=float(np.std(fv,ddof=1)/np.sqrt(5));ci=float(tdist.ppf(.975,4)*se);summary.append({"condition":c,"layer":l,"n":len(v),"mean_delta_margin":mean,"sd_item_delta_margin":float(np.std(v)),"ci95_low":mean-ci,"ci95_high":mean+ci,"two_sided_fold_t_p":float(ttest_1samp(fv,0).pvalue),"fold_means":" ".join(map(str,fv))})
        write_csv(self.out/"causal_summary.csv",summary);self.marker("aggregate",{"official":json.loads((self.out/"generator_selection.json").read_text())["official"]})
    def run(self, stages, resume=False):
        for stage in stages:
            for dep in DEPENDENCIES[stage]:
                if dep not in stages and not validate_stage(self.out,dep,self.fp):raise RuntimeError(f"Incomplete stage dependency: {stage} requires {dep}")
            if resume and validate_stage(self.out,stage,self.fp):continue
            getattr(self,stage)()
