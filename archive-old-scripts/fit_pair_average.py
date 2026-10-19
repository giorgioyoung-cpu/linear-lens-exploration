"""Fit per positional pair across passages, then average. No intercept or ridge.

Pilot: use existing complete 128-token training passages. Same-position and
pooled controls are refitted on exactly the same subset. Test data untouched.
Requires fit_affine.py alongside this file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from fit_affine import DEFAULT_RUN, sequence_moments, fit_linear, score


def average_pair_maps(U, Y, rtol=1e-10, progress=None):
    """U,Y: [passages, positions, dimension]. Return mean_{j<=i} A_ij.

    X_j has passage rows. A_ij=(pinv(X_j) @ Y_i).T. Linearity in Y_i
    permits summing targets BEFORE applying the shared pinv(X_j).
    rtol is an eigenvalue cutoff on S_j=X_j.T@X_j/N, matching prior fits;
    therefore the design singular-value cutoff is sqrt(rtol).
    """
    U, Y = np.asarray(U,dtype=np.float64), np.asarray(Y,dtype=np.float64)
    if U.ndim != 3 or U.shape != Y.shape or min(U.shape) < 1:
        raise ValueError('Expected matching nonempty [N,n,d] arrays.')
    if not np.isfinite(U).all() or not np.isfinite(Y).all():
        raise ValueError('Nonfinite residuals.')
    if not 0 <= rtol < 1:
        raise ValueError('rtol must be in [0,1).')
    N,n,d = U.shape
    total = np.zeros((d,d))
    suffix = np.zeros((N,d))
    diagnostics = []
    for j in range(n-1,-1,-1):
        suffix += Y[:,j,:]
        left,s,right = np.linalg.svd(U[:,j,:],full_matrices=False)
        keep = s > np.sqrt(rtol)*s[0]
        if keep.any():
            # Equivalent to summing all minimum-norm A_ij, i>=j.
            total += ((right[keep].T/s[keep]) @ (left[:,keep].T@suffix)).T
        info = {'position':j,'examples':N,'retained_rank':int(keep.sum()),
                'dimension':d,'largest_design_singular_value':float(s[0]),
                'smallest_retained_design_singular_value':float(s[keep][-1]) if keep.any() else None}
        diagnostics.append(info)
        if progress:
            progress(n-j,n,info)
    return total/(n*(n+1)/2), sorted(diagnostics,key=lambda v:v['position'])


def read_split(run, split, metadata, length, filter_short):
    import torch
    files = sorted((run/split).glob('sequence_*.pt'))
    if len(files) != metadata['splits'][split]['sequences'] or not files:
        raise ValueError(f'{split}: sequence count mismatch.')
    records, hashes, excluded = [], {}, []
    for path in files:
        raw = path.read_bytes()
        rec = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        U = rec['h_mid'].float().numpy()
        Y = rec['h_final'].float().numpy()
        n = len(U)
        if (U.shape != (n,metadata['hidden_size']) or Y.shape != U.shape
                or not np.isfinite(U).all() or not np.isfinite(Y).all()
                or rec['split'] != split or rec['layer'] != metadata['layer']
                or rec['final_layer'] != metadata['final_layer']
                or len(rec['input_ids']) != n
                or not np.array_equal(rec['positions'].numpy(),np.arange(n))):
            raise ValueError(f'Invalid record: {path}')
        if filter_short and n < length:
            excluded.append(path.name)
            continue
        if filter_short:
            U,Y = U[:length],Y[:length]
        records.append((U,Y))
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
    if not records:
        raise ValueError(f'No eligible {split} sequences of length {length}.')
    return records, {'sequences':len(records),'tokens':sum(len(u) for u,y in records),
                     'file_sha256':hashes,'excluded_short_files':excluded}


def moments(records, same=False):
    d = records[0][0].shape[1]
    S,C,q = np.zeros((d,d)),np.zeros((d,d)),0.
    for u,y in records:
        u,y = u.astype(np.float64),y.astype(np.float64)
        if same:
            s,c,t = u.T@u/len(u), y.T@u/len(u), float(np.sum(y*y)/len(u))
        else:
            s,c,t = sequence_moments(u,y)
        S+=s; C+=c; q+=t
    return S/len(records),C/len(records),q/len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--length',type=int,default=128)
    parser.add_argument('--pinv-rtol',type=float,default=1e-10)
    args = parser.parse_args()
    if args.length<1 or not 0<=args.pinv_rtol<1:
        parser.error('Require positive length and 0 <= pinv-rtol < 1.')
    run = args.run_dir.expanduser().resolve()
    meta_path = run/'metadata.json'
    meta = json.loads(meta_path.read_text())
    if meta.get('status')!='complete' or meta.get('format_version')!=1:
        raise ValueError('Expected complete format-version-1 activations.')
    started = perf_counter()
    train, train_info = read_split(run,'train',meta,args.length,True)
    print(f'Training: {len(train)} passages, {args.length} tokens each; '
          f'excluded {len(train_info["excluded_short_files"])} short passages.',flush=True)
    print(f'Each positional regression has at most rank {min(len(train),meta["hidden_size"])}/{meta["hidden_size"]}.',flush=True)
    print('Using minimum-norm pseudoinverses, no centering, intercept, or ridge.',flush=True)
    def progress(k,n,info):
        if k==1 or k%8==0 or k==n:
            print(f'Position fits: {k}/{n}; j={info["position"]}, rank={info["retained_rank"]}',flush=True)
    A,diagnostics = average_pair_maps(np.stack([u for u,y in train]),
                                    np.stack([y for u,y in train]),args.pinv_rtol,progress)
    print('Fitting matched pooled and same-position controls...',flush=True)
    causal_train,same_train = moments(train),moments(train,True)
    pooled,_,pool_info = fit_linear(causal_train[0],causal_train[1],args.pinv_rtol)
    same,_,same_info = fit_linear(same_train[0],same_train[1],args.pinv_rtol)
    # Baselines are matched to the corresponding training objective.
    pair_mean = np.mean([np.arange(1,len(y)+1)@y.astype(np.float64)/(len(y)*(len(y)+1)/2)
                         for u,y in train],axis=0)
    same_mean = np.mean([y.astype(np.float64).mean(axis=0) for u,y in train],axis=0)
    validation,val_info = read_split(run,'validation',meta,args.length,False)
    causal_val,same_val = moments(validation),moments(validation,True)
    maps = {'pair_average':A,'pooled':pooled,'same_position':same}
    metrics = {}
    for split,causal,diagonal in [('train',causal_train,same_train),('validation',causal_val,same_val)]:
        metrics[split] = {}
        for name,B in maps.items():
            metrics[split][name] = {'all_causal_pairs':score(B,causal),'same_position':score(B,diagonal)}
            print(f'{split} / {name}: causal error {score(B,causal)["mean_squared_euclidean_error"]:.6f}; '
                  f'same-position error {score(B,diagonal)["mean_squared_euclidean_error"]:.6f}',flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = run/f'pair_average_fit_{stamp}'
    out.mkdir(exist_ok=False)
    np.savez(out/'linear_lens.npz',A=A,pooled_A=pooled,same_position_A=same,
             pair_training_mean=pair_mean,same_training_mean=same_mean)
    mean_dir = out/f'mean_comparison_{stamp}'
    mean_dir.mkdir()
    np.save(mean_dir/'training_target_mean.npy',pair_mean)
    report = {'format_version':1,'created_utc':stamp,'activation_run':str(run),
        'collection_metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        'model_name':meta['model_name'],'model_revision':meta.get('model_revision'),
        'source_layer':meta['layer'],'target_layer':meta['final_layer'],
        'target':'final residual before normalization','intercept':False,'centered':False,'ridge':False,
        'pair_set':f'0 <= j <= i < {args.length}', 'sequence_length':args.length,
        'weighting':'uniform training passages for each pair; then uniform over all admissible pairs',
        'solver':'minimum-norm SVD; relative singular cutoff sqrt(pinv_rtol)',
        'pinv_rtol':args.pinv_rtol,'position_diagnostics':diagnostics,
        'controls':{'pooled':pool_info,'same_position':same_info},
        'metrics':metrics,'data':{'train':train_info,'validation':val_info},
        'test_evaluated':False,'elapsed_seconds':perf_counter()-started,
        'limitations':['Per-position fits may be underdetermined.',
                      'All causal pairs included; not a reproduction of original J-lens position exclusions.',
                      'Validation losses score the final averaged map, not the separate pair-specific maps.',
                      'Validation losses use full available sequence lengths.']}
    (out/'fit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'\nSaved fit to: {out}',flush=True)
    print('Next: python evaluate_pair_average.py',flush=True)


if __name__=='__main__':
    main()
