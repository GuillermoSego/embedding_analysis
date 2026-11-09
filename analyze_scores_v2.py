import numpy as np, sys
sys.path.insert(0,'.')
from analyze_hard_texts import load_train_labels
from scipy.stats import spearmanr
E_=["anger","fear","joy","sadness","surprise"]
V={};H={};L={};O={}
for e in E_:
    H[e]=np.load(f"eval_src_scores_v2/{e}/eval_src_scores_hits.npy"); V[e]=H[e].mean(0)
    L[e]=np.array([int(x) for x in load_train_labels(e)]); O[e]=np.load(f"eval_src_scores/{e}/eval_src_scores.npy")
    assert len(L[e])==H[e].shape[1]
print("emo  zeros  <.1  <.2  <.5  distinct  pos%all | pos%zeros | thr100 n<=thr pos%hard100 | accBase accEvo | splitRho | rhoOld")
rng=np.random.default_rng(5)
for e in E_:
    s=V[e];h=H[e];y=L[e]; z=s==0
    order=np.lexsort((rng.random(len(s)),s)); t100=s[order[99]]
    base=h[:10].mean(0); evo=h[10:].mean(0)
    print(f"{e:9} {z.sum():4} {(s<.1).sum():4} {(s<.2).sum():4} {(s<.5).sum():4} {len(np.unique(s.round(4))):3}  {y.mean()*100:4.1f} | {y[z].mean()*100:5.1f} | {t100:.3f} {(s<=t100).sum():4} {y[order[:100]].mean()*100:5.1f} | {h[:10].mean():.3f} {h[10:].mean():.3f} | {spearmanr(base,evo)[0]:.2f} | {spearmanr(s,O[e])[0]:.2f}")
print("\nper-prompt acc min/med/max")
for e in E_:
    a=H[e].mean(1); print(e, f"{a.min():.2f}/{np.median(a):.2f}/{a.max():.2f}")
print("\noverlap old hard100 vs new lowest100")
for e in E_:
    o=O[e]; idx=np.where(o>0.001)[0]; old=set(idx[np.argsort(o[idx],kind='stable')[:100]])
    s=V[e]; new=set(np.lexsort((rng.random(len(s)),s))[:100]); z=set(np.where(s==0)[0])
    print(e,"overlap",len(old&new),"| old set mean new-score",round(float(np.mean([s[i] for i in old])),3),"| old∩newzeros",len(old&z))
Z=np.stack([V[e]==0 for e in E_]); print("\ntexts zero in k emotions:",np.bincount(Z.sum(0),minlength=6))
print(np.corrcoef(np.stack([V[e] for e in E_])).round(2))
