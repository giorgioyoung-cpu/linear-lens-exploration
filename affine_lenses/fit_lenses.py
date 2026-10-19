"""Fit pooled, same and paired affine slopes on identical full-length training passages."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import platform
import numpy as np
from lens_core import DEFAULT_RUN,NAMES,stamp,sha,stats,fit_stats,fit_paired,error,read_records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--pinv-rtol',type=float,default=1e-10)
    args=p.parse_args()
    if args.length<=17 or not 0<args.pinv_rtol<1: p.error('length > 17 and 0 < pinv-rtol < 1 required.')
    start=perf_counter()
    run=args.run_dir.expanduser().resolve()
    meta=json.loads((run/'metadata.json').read_text())
    if meta.get('status')!='complete' or meta.get('format_version')!=1: raise ValueError('Incomplete/unsupported collection.')
    train,ti=read_records(run,'train',meta,args.length)
    val,vi=read_records(run,'validation',meta)
    print(f'Train: {len(train)} full-length passages; excluded {len(ti["excluded_short_files"])} short passages.',flush=True)
    print(f'One-based training positions: 17–{args.length-1}. Paired rank <= {min(len(train)-1,meta["hidden_size"])}.',flush=True)
    train_stats={name:stats(train,name=='same') for name in ('pooled','same')}
    val_stats={name:stats(val,name=='same') for name in ('pooled','same')}
    arrays,diagnostics={},{}
    for name,s in train_stats.items():
        A,b,info=fit_stats(s,args.pinv_rtol)
        arrays.update({name+'_A':A,name+'_b':b,name+'_mean':s[4]})
        diagnostics[name]=info
        print(f'{name}: retained rank {info["rank"]}/{info["dimension"]}',flush=True)
    def progress(k,n,info):
        if k==1 or k%8==0 or k==n: print(f'paired: {k}/{n}; position {info["position"]}; rank {info["rank"]}',flush=True)
    A,b,info=fit_paired(np.stack([u for u,y in train]),np.stack([y for u,y in train]),args.pinv_rtol,progress)
    arrays.update(paired_A=A,paired_b=b)
    diagnostics['paired']=info
    metrics={}
    for split,ss in [('train',train_stats),('validation',val_stats)]:
        metrics[split]={}
        for name in NAMES:
            A,b=arrays[name+'_A'],arrays[name+'_b']
            metrics[split][name]={}
            for objective,s in ss.items():
                baseline=error(np.zeros_like(A),train_stats[objective][4],s)
                slope=error(A,np.zeros_like(b),s)
                full=error(A,b,s)
                metrics[split][name][objective]={'slope_only_mse':slope,'with_intercept_mse':full,
                    'training_mean_mse':baseline,'identity_mse':error(np.eye(len(b)),np.zeros_like(b),s),
                    'slope_reduction_vs_mean':1-slope/baseline if baseline else None,
                    'affine_reduction_vs_mean':1-full/baseline if baseline else None}
            print(f'{split}/{name}: same-position slope MSE {metrics[split][name]["same"]["slope_only_mse"]:.6f}; with intercept {metrics[split][name]["same"]["with_intercept_mse"]:.6f}',flush=True)
    out=run/('affine_fit_'+stamp()); out.mkdir()
    np.savez(out/'lenses.npz',**arrays)
    report={'format_version':2,'activation_run':str(run),'collection_metadata_sha256':sha(run/'metadata.json'),
        'model_name':meta['model_name'],'model_revision':meta.get('model_revision'),'source_layer':meta['layer'],
        'target_layer':meta['final_layer'],'hidden_size':meta['hidden_size'],'centered':True,'intercept_fitted':True,
        'readout':'slope only; intercept discarded','ridge':False,'training_length':args.length,
        'mask_one_based':'17 <= j <= i <= n-1','pinv_rtol':args.pinv_rtol,
        'solver':'eigenvalue cutoff rtol for pooled/same; design SVD cutoff sqrt(rtol) for paired',
        'data':{'train':ti,'validation':vi},'diagnostics':diagnostics,'metrics':metrics,'test_used':False,
        'elapsed_seconds':perf_counter()-start,'python_version':platform.python_version(),'numpy_version':np.__version__,
        'notes':['All three fitted on identical fixed-length training passages.',
                 'Validation uses all available passages with its own 17..n-1 mask.',
                 'paired_b is the average of pair intercepts, not the optimal intercept of the averaged slope.',
                 'Per-pair intercepts/slopes are aggregated exactly without materializing every pair matrix.',
                 'J-lens has different sample count and sum-target scaling; mask matching alone does not match estimators.']}
    (out/'fit_report.json').write_text(json.dumps(report,indent=2))
    print(f'Saved: {out}\nNext: python evaluate_lenses.py --fit-dir "{out}"',flush=True)

if __name__=='__main__': main()
