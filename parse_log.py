import re,sys,numpy as np
from scipy.stats import wilcoxon
L=open(sys.argv[1]).read().splitlines()
runs={};cur=[]
for l in L:
    m=re.search(r'(S0\d) \[loso\].*online-EA ([\d.]+) \(k=([-\d.]+)\) \| oracle ([\d.]+) \| no-EA ([\d.]+)',l)
    if m: cur.append([float(m.group(i)) for i in (2,3,4,5)])
    m=re.search(r'== (\w+) done',l)
    if m: runs[m.group(1)]=np.array(cur) if cur else None; cur=[]
R={k:v for k,v in runs.items() if v is not None and len(v)==9}
for k,v in R.items(): print(f'{k:16s} online {v[:,0].mean()*100:.2f} k {v[:,1].mean():.3f} oracle {v[:,2].mean()*100:.2f} noEA {v[:,3].mean()*100:.2f} |on-or| {np.abs(v[:,0]-v[:,2]).mean()*100:.2f}')
def grp(p): 
    A=np.stack([R[f'{p}_s{s}'] for s in (42,7,123) if f'{p}_s{s}' in R]); return A
for p in ('A4_prop','A4_eegnet','A2_prop','A2_eegnet'):
    A=grp(p); sm=A[:,:,0].mean(1)*100
    print(p,len(A),'seeds; online',f'{sm.mean():.2f} ± {sm.std():.2f}','subj-std',f'{(A[:,:,0].mean(0)*100).std():.2f}','k',f'{A[:,:,1].mean():.3f}',
      'oracle',f'{A[:,:,2].mean()*100:.2f}','noEA',f'{A[:,:,3].mean()*100:.2f}','per-subj',np.round(A[:,:,0].mean(0)*100,1))
for a,b in (('A4_prop','A4_eegnet'),('A2_prop','A2_eegnet')):
    x=grp(a)[:,:,0].mean(0); y=grp(b)[:,:,0].mean(0); print(a,'vs',b,f'diff {(x-y).mean()*100:+.2f} p={wilcoxon(x,y).pvalue:.3f}')
# online vs oracle over all runs, paired subject-level
on=np.concatenate([v[:,0] for v in R.values()]); orc=np.concatenate([v[:,2] for v in R.values()]); ne=np.concatenate([v[:,3] for v in R.values()])
print('all folds',len(on),'online-oracle mean',f'{(on-orc).mean()*100:+.2f}','p',wilcoxon(on,orc).pvalue,'online-noEA',f'{(on-ne).mean()*100:+.2f}',wilcoxon(on,ne).pvalue)
for p in ('A4_prop','A2_prop'):
    A=grp(p); x=A[:,:,0].mean(0); o=A[:,:,2].mean(0); n=A[:,:,3].mean(0)
    print(p,'online vs oracle p',f'{wilcoxon(x,o).pvalue:.3f}','vs noEA p',f'{wilcoxon(x,n).pvalue:.4f}')
