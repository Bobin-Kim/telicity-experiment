from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from .preflight import build_prompt, folds
from .output_validation import atomic_json, write_csv, validate_stage

DEPENDENCIES={"behavior":[],"cache":[],"predictive_localization":["cache"],"causal_intervention":["cache","predictive_localization"],"component_analysis":["cache"],"generator_localization":["component_analysis","causal_intervention"],"aggregate":["behavior","predictive_localization","causal_intervention","generator_localization"],"upstream_source":["cache","generator_localization"],"reemergence_cache":[],"reemergence_sweep":["reemergence_cache"],"source_pathway":["reemergence_sweep"],"source_pathway_extended":["reemergence_sweep"],"reemergence_aggregate":["source_pathway"]}

def ridge(x,y,a=10.):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1; z=(x-mu)/sd; yc=y-y.mean()
    c=np.linalg.solve(z.T@z+a*np.eye(z.shape[1]),z.T@yc) if z.shape[1]<=len(z) else z.T@np.linalg.solve(z@z.T+a*np.eye(len(z)),yc)
    return mu,sd,c,float(y.mean())
def pred(f,x): mu,sd,c,b=f; return ((x-mu)/sd)@c+b
def r2(y,p): return 1-float(((y-p)**2).sum())/max(float(((y-y.mean())**2).sum()),1e-12)
def norm(x): return x/max(float(np.linalg.norm(x)),1e-12)

class PipelineController:
    def __init__(self, model, tokenizer, adapter, scorer, rows, fixed, out, fingerprint, batch_size=None, layers=None, heldout_folds=None):
        self.model,self.tok,self.a,self.s,self.rows,self.f,self.out,self.fp=model,tokenizer,adapter,scorer,rows,fixed,Path(out),fingerprint
        self.batch_override=batch_size;self.compat=adapter.spec.get("protocol_compatibility",{});self.profile=self.compat.get("profile")
        self.fold=np.asarray(folds(rows,fixed)); self.device=next(model.parameters()).device
        self.requested_layers=list(range(adapter.num_layers)) if layers is None else layers
        self.requested_folds=list(range(fixed["folds"]["count"])) if heldout_folds is None else heldout_folds
        if any(not 0<=x<adapter.num_layers for x in self.requested_layers):raise ValueError("Invalid requested layer")
        if any(not 0<=x<fixed["folds"]["count"] for x in self.requested_folds):raise ValueError("Invalid requested fold")
    def stage_batch(self,stage):
        if self.batch_override is not None:return self.batch_override
        return int(self.compat.get(f"{stage}_batch_size",4))
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
        rows=[];batch=self.stage_batch("behavior")
        groups=[np.flatnonzero(self.fold==f).tolist() for f in range(5)] if self.profile=="mistral_original_rq2_e345" else [list(range(len(self.rows)))]
        for group in groups:
          for st in range(0,len(group),batch):
            ids=group[st:st+batch]; z,_,_=self.forward(ids)
            for j,i in enumerate(ids): rows.append({"example_index":i,"example_id":self.rows[i]["id"],"group":self.rows[i]["group"][0],"fold":int(self.fold[i]),"true_logit":float(z["true_logit"][j]),"unknown_logit":float(z["unknown_logit"][j]),"margin":float(z["margin"][j])})
        rows.sort(key=lambda r:r["example_index"])
        write_csv(self.out/"behavior.csv",rows); self.marker("behavior",{"rows":len(rows),"scoring_mode":self.s.mode})
    def cache(self):
        R,A,M,H,margin=[],[],[],[],[]; L=self.a.num_layers;batch=self.stage_batch("representation")
        for st in range(0,len(self.rows),batch):
            inds=list(range(st,min(st+batch,len(self.rows)))); e=self.encode(inds); pos=self.s.positions(e["attention_mask"]); cap={"r":[None]*L,"a":[None]*L,"m":[None]*L,"h":[None]*L}; handles=[]
            def pre(k):
                def fn(_m,inp): cap["r"][k]=inp[0].detach()
                return fn
            def post(kind,k):
                def fn(_m,_i,out): cap[kind][k]=(out[0] if isinstance(out,tuple) else out).detach()
                return fn
            def hpre(k):
                def fn(_m,inp): cap["h"][k]=inp[0].detach()
                return fn
            def residual_post(k):
                def fn(_m,_i,out):cap["r"][k]=(out[0] if isinstance(out,tuple) else out).detach()
                return fn
            for l in range(L):
                if self.compat.get("residual_coordinate")=="block_output_with_final_norm_at_last_layer":
                    if l<L-1:handles.append(self.a.layers[l].register_forward_hook(residual_post(l)))
                else:handles.append(self.a.layers[l].register_forward_pre_hook(pre(l)))
                handles += [self.a.attention(l).register_forward_hook(post("a",l)),self.a.mlp(l).register_forward_hook(post("m",l)),self.a.head_input(l).register_forward_pre_hook(hpre(l))]
            if self.compat.get("residual_coordinate")=="block_output_with_final_norm_at_last_layer":handles.append(self.a.final_norm.register_forward_hook(residual_post(L-1)))
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
        target_margin=np.concatenate(margin)
        if self.profile=="mistral_original_rq2_e345":
            # The original decision/localization stage consumed this frozen
            # first-token margin artifact from the completed Mistral fold run;
            # it did not recompute margins in the localization process.
            source=Path(self.compat["canonical_margin"]);target_margin=np.load(source,allow_pickle=False).astype(np.float32)
            if target_margin.shape!=(len(self.rows),) or not np.isfinite(target_margin).all():raise ValueError(f"Invalid canonical Mistral margin artifact: {source}")
        arrays={"residual":np.concatenate(R),"attention":np.concatenate(A),"mlp":np.concatenate(M),"head":np.concatenate(H),"margin":target_margin,"fold":self.fold}
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError("NaN/Inf in component cache")
        np.savez_compressed(self.out/"preference_cache.npz",**arrays)
        self.marker("cache",{"rows":len(self.rows),"layers":L,"layer_coverage":list(range(L)),"position_policy":self.f["scoring_position"],"residual_coordinate":self.compat.get("residual_coordinate","block_input"),"target_margin_source":self.compat.get("canonical_margin","fresh shared scorer")})
    def predictive_localization(self):
        if self.profile=="mistral_original_rq2_e345":return self._mistral_predictive_localization()
        z=np.load(self.out/"preference_cache.npz"); x=z["residual"].astype(float); y=z["margin"].astype(float); records=[]; selected=[]
        for held in self.requested_folds:
            tr=self.fold!=held; te=~tr; scores=[]
            for l in range(self.a.num_layers):
                p=pred(ridge(x[tr,l],y[tr],self.f["preference_basis"]["ridge_alpha"]),x[te,l]); score=r2(y[te],p); records.append({"fold":held,"layer":l,"heldout_r2":score}); inner=[]
                for v in range(self.f["folds"]["count"]):
                    it=tr&(self.fold!=v); va=tr&(self.fold==v)
                    if va.any():inner.append(r2(y[va],pred(ridge(x[it,l],y[it]),x[va,l])))
                scores.append((float(np.mean(inner)),l))
            best=max(scores); selected.append({"heldout_fold":held,"layer":best[1],"inner_cv_r2":best[0],"selection":"all-layer inner-CV; no prior layer"})
        write_csv(self.out/"predictive_full_depth.csv",records); write_csv(self.out/"selected_layers.csv",selected); self.marker("predictive_localization",{"layers":self.a.num_layers,"folds":self.requested_folds,"official_fold_coverage":len(self.requested_folds)==self.f["folds"]["count"]})
    def _mistral_predictive_localization(self):
        from sklearn.linear_model import Ridge
        from scipy.stats import pearsonr,spearmanr
        z=np.load(self.out/"preference_cache.npz");x=z["residual"].astype(np.float32);y=z["margin"].astype(np.float32);L=self.a.num_layers;dirs=np.zeros((5,L,x.shape[-1]),np.float32);pred_rows=[];fold_rows=[];summary=[]
        for layer in range(L):
            # Match the original script's default NumPy float (float64) OOF
            # buffer; Ridge inputs/targets remain float32 as in the source.
            oof=np.zeros(len(y),float)
            for held in self.requested_folds:
                tr=self.fold!=held;te=~tr;mu=x[tr,layer].mean(0);sd=x[tr,layer].std(0);sd[sd<1e-6]=1
                fit=Ridge(alpha=float(self.f["preference_basis"]["ridge_alpha"])).fit((x[tr,layer]-mu)/sd,y[tr]);oof[te]=fit.predict((x[te,layer]-mu)/sd);v=fit.coef_/sd;dirs[held,layer]=v/max(float(np.linalg.norm(v)),1e-12)
                score=r2(y[te].astype(float),oof[te].astype(float));fold_rows.append({"fold":held,"layer":layer,"heldout_r2":score})
                for i in np.flatnonzero(te):pred_rows.append({"layer":layer,"fold":held,"example_id":self.rows[i]["id"],"margin":float(y[i]),"predicted_margin":float(oof[i])})
            evaluated=np.isin(self.fold,self.requested_folds)
            rr=r2(y[evaluated].astype(float),oof[evaluated].astype(float));pr=float(pearsonr(y[evaluated],oof[evaluated]).statistic);sr=float(spearmanr(y[evaluated],oof[evaluated]).statistic);summary.append({"layer":layer,"r2":rr,"pearson_r":pr,"spearman_rho":sr})
        np.save(self.out/"decision_directions.npy",dirs);write_csv(self.out/"predictive_full_depth.csv",fold_rows);write_csv(self.out/"localization_predictions.csv",pred_rows);write_csv(self.out/"localization_summary.csv",summary)
        best=max(summary,key=lambda q:q["r2"]);selected=[{"heldout_fold":f,"layer":best["layer"],"oof_r2":best["r2"],"selection":"Mistral OOF strongest rank-1 ridge layer over requested folds"} for f in self.requested_folds];write_csv(self.out/"selected_layers.csv",selected)
        stable=[q["layer"] for q in summary if q["r2"]>.05 and q["pearson_r"]>.25];self.marker("predictive_localization",{"layers":L,"folds":self.requested_folds,"official_fold_coverage":len(self.requested_folds)==5,"axis_estimator":"standardized_rank1_ridge","first_stable_layer":min(stable) if stable else None,"strongest_layer":best["layer"]})
    def _axis(self,x,y,tr):
        center=x[tr].mean(0); work=x.copy(); basis=[]
        for _ in range(self.f["preference_basis"]["components"]):
            mu,sd,c,_=ridge(work[tr],y[tr]); v=c/sd
            for q in basis:v-=q*np.dot(q,v)
            if np.linalg.norm(v)<1e-10:break
            v=norm(v);basis.append(v);work-=np.outer(work@v,v)
        b=np.stack(basis); axis=norm(b.T@np.linalg.lstsq((x[tr]-center)@b.T,y[tr]-y[tr].mean(),rcond=1e-8)[0]); return center,b,axis,float(np.std((x[tr]-center)@axis) or 1)
    def causal_intervention(self):
        if self.profile=="mistral_original_rq2_e345":return self._mistral_causal_intervention()
        z=np.load(self.out/"preference_cache.npz"); x=z["residual"].astype(float); y=z["margin"].astype(float); rows=[]; seed=self.f["random_seed"]
        for held in self.requested_folds:
            tr=self.fold!=held; test=np.flatnonzero(~tr).tolist()
            for layer in range(self.a.num_layers):
                center,basis,axis,sd=self._axis(x[:,layer],y,tr); rng=np.random.default_rng(seed+held*1000+layer); raw=rng.normal(size=(x.shape[-1],len(basis)));raw-=basis.T@(basis@raw);rbase=np.linalg.qr(raw)[0].T;rv=rbase[0];decision_dist=np.linalg.norm(((x[tr,layer]-center)@basis.T)@basis,axis=1).mean();random_dist=np.linalg.norm(((x[tr,layer]-center)@rbase.T)@rbase,axis=1).mean();rscale=float(decision_dist/max(random_dist,1e-12))
                for cond in self.f["intervention"]["conditions"]:
                    batch=self.stage_batch("intervention")
                    for st in range(0,len(test),batch):
                        inds=test[st:st+batch]; e=self.encode(inds); pos=self.s.positions(e["attention_mask"])
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
        for held in self.requested_folds:
            for layer in range(self.a.num_layers):
                shard=[r for r in rows if r["heldout_fold"]==held and r["layer"]==layer];d=self.out/"stage_a"/f"L{layer:02d}"/f"fold{held}";write_csv(d/"interventions.csv",shard);atomic_json(d/"complete.json",{"status":"complete","fold":held,"layer":layer,"rows":len(shard),"run_fingerprint":self.fp})
        self.marker("causal_intervention",{"rows":len(rows),"layers":self.a.num_layers,"folds":self.requested_folds,"official_fold_coverage":len(self.requested_folds)==self.f["folds"]["count"],"conditions":self.f["intervention"]["conditions"]})
    def _mistral_controls(self,hidden,dirs):
        rng=np.random.default_rng(self.f["random_seed"]+int(self.compat["control_seed_offset"]));L=self.a.num_layers;random_dirs=np.zeros_like(dirs,np.float32);scales=np.zeros((5,L),np.float32);means=np.zeros((5,L,hidden.shape[-1]),np.float32);floor=float(self.compat["scale_floor"])
        for held in self.requested_folds:
            tr=self.fold!=held
            for layer in range(L):
                v=dirs[held,layer].astype(np.float64);v/=np.linalg.norm(v);r=rng.normal(size=len(v));r-=v*(r@v);r/=np.linalg.norm(r);random_dirs[held,layer]=r;x=hidden[tr,layer].astype(np.float32);means[held,layer]=x.mean(0);xc=x-means[held,layer];sds=[]
                for _ in range(16):
                    q=rng.normal(size=len(v));q-=v*(q@v);q/=np.linalg.norm(q);sds.append(float(np.std(xc@q)))
                scales[held,layer]=max(float(np.median(sds)),floor)
        return random_dirs,scales,means
    def _mistral_causal_intervention(self):
        z=np.load(self.out/"preference_cache.npz");hidden=z["residual"].astype(np.float32);dirs=np.load(self.out/"decision_directions.npy").astype(np.float32);random_dirs,scales,means=self._mistral_controls(hidden,dirs);np.savez_compressed(self.out/"train_only_controls.npz",random_dirs=random_dirs,scales=scales,means=means)
        import csv
        baseline={r["example_id"]:float(r["margin"]) for r in csv.DictReader(open(self.out/"behavior.csv"))};conditions=tuple(self.compat["intervention_conditions"]);batch=self.stage_batch("intervention");rows=[];L=self.a.num_layers
        for layer in self.requested_layers:
            for held in self.requested_folds:
                test=np.flatnonzero(self.fold==held).tolist();v=dirs[held,layer].astype(np.float64);v/=np.linalg.norm(v);rv=random_dirs[held,layer].astype(np.float64);rv/=np.linalg.norm(rv);spec=(("decision_pos",v,1.),("decision_neg",v,-1.),("random_pos",rv,1.),("random_neg",rv,-1.));delta_axes=torch.tensor(np.stack([sign*axis for _,axis,sign in spec]),device=self.device,dtype=torch.float32)*float(scales[held,layer]);shard=[]
                for st in range(0,len(test),batch):
                    inds=test[st:st+batch];e0=self.encode(inds);a0=self.s.positions(e0["attention_mask"]);n=len(spec);bs=len(inds);enc={k:v.repeat((n,1)) for k,v in e0.items()};pos=a0.repeat(n);deltas=delta_axes.repeat_interleave(bs,dim=0)
                    def hook(_m,_i,out):
                        value=(out[0] if isinstance(out,tuple) else out).clone();ix=torch.arange(n*bs,device=value.device);value[ix,pos]=(value[ix,pos].float()+deltas).to(value.dtype);return (value,*out[1:]) if isinstance(out,tuple) else value
                    module=self.a.final_norm if layer==L-1 else self.a.layers[layer];handle=module.register_forward_hook(hook)
                    try:
                        with torch.inference_mode():o=self.model(**enc,use_cache=False,return_dict=True)
                    finally:handle.remove()
                    scored=self.s.from_logits(o.logits,pos)["margin"].cpu().numpy()
                    for ci,(name,_axis,_sign) in enumerate(spec):
                        for j,i in enumerate(inds):
                            q=ci*bs+j;m=float(scored[q]);rec={"layer":layer,"fold":held,"heldout_fold":held,"example_index":i,"example_id":self.rows[i]["id"],"group":self.rows[i]["group"][0],"condition":name,"baseline_margin":baseline[self.rows[i]["id"]],"margin":m,"delta_margin":m-baseline[self.rows[i]["id"]],"shared_norm":float(scales[held,layer])};rows.append(rec);shard.append(rec)
                d=self.out/"stage_a"/f"L{layer:02d}"/f"fold{held}";write_csv(d/"interventions.csv",shard);atomic_json(d/"complete.json",{"status":"complete","layer":layer,"fold":held,"rows":len(shard),"conditions":len(conditions),"hook_coordinate":"final norm" if layer==L-1 else f"decoder block {layer} output","run_fingerprint":self.fp})
        write_csv(self.out/"causal_full_depth.csv",rows);self.marker("causal_intervention",{"rows":len(rows),"layers":self.requested_layers,"folds":self.requested_folds,"full_coverage":self.requested_layers==list(range(L)) and self.requested_folds==list(range(5)),"conditions":list(conditions),"axis_estimator":self.compat["axis_estimator"],"control_scale":self.compat["control_scale"]})
    def component_analysis(self):
        z=np.load(self.out/"preference_cache.npz"); y=z["margin"].astype(float); out=[]; heads=[]
        for comp in ("residual","attention","mlp"):
            x=z[comp].astype(float)
            for held in self.requested_folds:
                tr=self.fold!=held;te=~tr
                for l in range(self.a.num_layers):
                    inner=[]
                    for v in range(5):
                        it=tr&(self.fold!=v);va=tr&(self.fold==v)
                        if va.any():inner.append(r2(y[va],pred(ridge(x[it,l],y[it]),x[va,l])))
                    out.append({"component":comp,"fold":held,"layer":l,"heldout_r2":r2(y[te],pred(ridge(x[tr,l],y[tr]),x[te,l])),"inner_cv_r2":float(np.mean(inner))})
        hx=z["head"].astype(float).reshape(len(y),self.a.num_layers,self.a.num_heads,self.a.head_dim)
        for held in self.requested_folds:
            tr=self.fold!=held;te=~tr
            for l in range(self.a.num_layers):
                for h in range(self.a.num_heads):heads.append({"fold":held,"layer":l,"head":h,"heldout_r2":r2(y[te],pred(ridge(hx[tr,l,h],y[tr]),hx[te,l,h]))})
        write_csv(self.out/"component_full_depth.csv",out);write_csv(self.out/"head_full_depth.csv",heads)
        # Candidate layers are selected independently inside each outer fold, then tested by actual forward intervention.
        causal=[]
        for held in self.requested_folds:
            tr=self.fold!=held;test=np.flatnonzero(~tr).tolist()
            for comp in ("attention","mlp"):
                candidates=[r for r in out if r["component"]==comp and r["fold"]==held];layer=max(candidates,key=lambda q:q["inner_cv_r2"])["layer"]
                x=z[comp][:,layer].astype(float);center=x[tr].mean(0);axis=norm((x[tr]-center).T@(y[tr]-y[tr].mean()));sd=float(np.std((x[tr]-center)@axis) or 1)
                rng=np.random.default_rng(self.f["random_seed"]+held*100+layer);rv=rng.normal(size=x.shape[-1]);rv-=axis*np.dot(axis,rv);rv=norm(rv)
                module=self.a.attention(layer) if comp=="attention" else self.a.mlp(layer)
                for cond in ("decision_pos","decision_neg","random_pos","random_neg"):
                    batch=self.stage_batch("intervention")
                    for st in range(0,len(test),batch):
                        inds=test[st:st+batch];e=self.encode(inds);pos=self.s.positions(e["attention_mask"])
                        def hook(_m,_inp,output):
                            base=output[0] if isinstance(output,tuple) else output;v=base.clone();ix=torch.arange(len(inds),device=v.device);direction=rv if cond.startswith("random") else axis;sign=1 if cond.endswith("pos") else -1;v[ix,pos]+=sign*sd*torch.tensor(direction,device=v.device,dtype=v.dtype);return (v,*output[1:]) if isinstance(output,tuple) else v
                        handle=module.register_forward_hook(hook)
                        try:
                            with torch.inference_mode():o=self.model(**e,use_cache=False,return_dict=True)
                        finally:handle.remove()
                        m=self.s.from_logits(o.logits,pos)["margin"].cpu().numpy()
                        for j,i in enumerate(inds):causal.append({"component":comp,"layer":layer,"head":-1,"heldout_fold":held,"example_id":self.rows[i]["id"],"condition":cond,"baseline_margin":float(y[i]),"margin":float(m[j]),"delta_margin":float(m[j]-y[i])})
        write_csv(self.out/"component_causal.csv",causal)
        for held in self.requested_folds:
            shard=[r for r in causal if r["heldout_fold"]==held];d=self.out/"stage_b_onset"/f"fold{held}";write_csv(d/"interventions.csv",shard);entities={(r["component"],r["layer"],r["head"]) for r in shard};atomic_json(d/"complete.json",{"status":"complete","fold":held,"rows":len(shard),"entities":len(entities),"run_fingerprint":self.fp})
        self.marker("component_analysis",{"components":["residual","attention","mlp","heads"],"layers":self.a.num_layers,"folds":self.requested_folds,"official_fold_coverage":len(self.requested_folds)==self.f["folds"]["count"],"causal_controls":["decision_pos","decision_neg","random_pos","random_neg"]})
    def generator_localization(self):
        import csv
        from scipy.stats import t as tdist, ttest_1samp
        with (self.out/"component_causal.csv").open() as f: rows=list(csv.DictReader(f))
        summary=[]
        for comp in ("attention","mlp"):
            fold_effect=[];layers=[]
            for held in self.requested_folds:
                z=[r for r in rows if r["component"]==comp and int(r["heldout_fold"])==held];layers.append(int(z[0]["layer"]));pos=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="decision_pos"]);neg=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="decision_neg"]);rp=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="random_pos"]);rn=np.mean([float(r["delta_margin"]) for r in z if r["condition"]=="random_neg"]);fold_effect.append((pos-neg)-(rp-rn))
            mean=float(np.mean(fold_effect));n=len(fold_effect)
            if n>1:
                se=float(np.std(fold_effect,ddof=1)/np.sqrt(n));ci=float(tdist.ppf(.975,n-1)*se);p=float(ttest_1samp(fold_effect,0).pvalue);lo,hi=mean-ci,mean+ci
            else:
                p=lo=hi=""
            summary.append({"component":comp,"median_selected_layer":int(np.median(layers)),"mean_control_adjusted_contrast":mean,"ci95_low":lo,"ci95_high":hi,"two_sided_fold_t_p":p,"positive_folds":sum(x>0 for x in fold_effect),"evaluated_folds":n})
        full=len(self.requested_folds)==5
        reliable=[r for r in summary if full and r["positive_folds"]>=4 and float(r["ci95_low"])>0]; chosen=min(reliable,key=lambda r:(r["median_selected_layer"],0 if r["component"]=="attention" else 1)) if reliable else None
        status="complete" if chosen else ("audit_failure" if full else "exploratory_complete")
        write_csv(self.out/"generator_candidates.csv",summary); selection={"status":status,"official":bool(chosen),"earliest_reliable_component":chosen,"requested_folds":self.requested_folds,"criterion":"official only with all 5 folds: actual forward decision contrast minus matched random contrast; 95% fold-t CI > 0 and >=4/5 positive folds; no prior component"};atomic_json(self.out/"generator_selection.json",selection);atomic_json(self.out/"stage_b_selection.json",selection);self.marker("generator_localization",{"official":bool(chosen),"folds":self.requested_folds})
    def upstream_source(self):
        selection=json.loads((self.out/"generator_selection.json").read_text())["earliest_reliable_component"]
        if not selection: raise RuntimeError("Generator audit failed; upstream source is non-interpretable")
        z=np.load(self.out/"preference_cache.npz"); y=z["margin"].astype(float); rows=[]
        for comp in ("attention","mlp"):
            for l in range(int(selection["median_selected_layer"])):
                x=z[comp][:,l].astype(float); rows.append({"component":comp,"layer":l,"correlation":float(np.corrcoef(x@norm((x-x.mean(0)).T@(y-y.mean())),y)[0,1])})
        write_csv(self.out/"upstream_source.csv",rows);self.marker("upstream_source",{"supplemental":True})
    def reemergence_cache(self):
        from .reemergence import ReemergenceWorkflow
        ReemergenceWorkflow(self).cache()
    def reemergence_sweep(self):
        from .reemergence import ReemergenceWorkflow
        ReemergenceWorkflow(self).sweep()
    def source_pathway(self):
        from .reemergence import ReemergenceWorkflow
        ReemergenceWorkflow(self).pathways(extended=False)
    def source_pathway_extended(self):
        from .reemergence import ReemergenceWorkflow
        ReemergenceWorkflow(self).pathways(extended=True)
    def reemergence_aggregate(self):
        from .reemergence import ReemergenceWorkflow
        ReemergenceWorkflow(self).aggregate()
    def aggregate(self):
        import csv
        from scipy.stats import t as tdist, ttest_1samp
        with (self.out/"causal_full_depth.csv").open() as f: r=list(csv.DictReader(f)); summary=[]
        for c in self.f["intervention"]["conditions"]:
            for l in range(self.a.num_layers):
                v=[float(x["delta_margin"]) for x in r if x["condition"]==c and int(x["layer"])==l];fv=[np.mean([float(x["delta_margin"]) for x in r if x["condition"]==c and int(x["layer"])==l and int(x["heldout_fold"])==f]) for f in self.requested_folds];mean=float(np.mean(fv));nfold=len(fv)
                if nfold>1:
                    se=float(np.std(fv,ddof=1)/np.sqrt(nfold));ci=float(tdist.ppf(.975,nfold-1)*se);lo,hi,p=mean-ci,mean+ci,float(ttest_1samp(fv,0).pvalue)
                else:lo=hi=p=""
                summary.append({"condition":c,"layer":l,"n":len(v),"mean_delta_margin":mean,"sd_item_delta_margin":float(np.std(v)),"ci95_low":lo,"ci95_high":hi,"two_sided_fold_t_p":p,"fold_means":" ".join(map(str,fv)),"evaluated_folds":nfold})
        write_csv(self.out/"causal_summary.csv",summary);self.marker("aggregate",{"official":len(self.requested_folds)==5 and json.loads((self.out/"generator_selection.json").read_text())["official"],"folds":self.requested_folds})
    def run(self, stages, resume=False):
        for stage in stages:
            for dep in DEPENDENCIES[stage]:
                if dep not in stages and not validate_stage(self.out,dep,self.fp):raise RuntimeError(f"Incomplete stage dependency: {stage} requires {dep}")
            if resume and validate_stage(self.out,stage,self.fp):continue
            getattr(self,stage)()
