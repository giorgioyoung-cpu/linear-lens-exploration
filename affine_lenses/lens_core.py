"""NumPy affine fits. Matrices map column residuals; no ridge penalty."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

DEFAULT_RUN = Path.home()/'jspace-data/activations/layer12_20261014T191345_397425Z'
NAMES = ('pooled', 'same', 'paired')

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(records, same=False):
    """Equal passage weight, uniform retained pairs (or diagonal positions).

    Returns raw E[uu^T], E[yu^T], E[||y||^2], E[u], E[y].
    Records must already have the positional mask applied.
    """
    d = records[0][0].shape[1]
    S,C,q,mu,my = np.zeros((d,d)),np.zeros((d,d)),0.,np.zeros(d),np.zeros(d)
    for u,y in records:
        u,y = np.asarray(u,dtype=np.float64),np.asarray(y,dtype=np.float64)
        n = len(u)
        if not n or u.shape != y.shape or not np.isfinite(u).all() or not np.isfinite(y).all():
            raise ValueError('Invalid residual arrays.')
        a = np.ones(n) if same else np.arange(n,0,-1)
        b = np.ones(n) if same else np.arange(1,n+1)
        m = a.sum()
        target = y if same else np.cumsum(y[::-1],axis=0)[::-1]
        S += u.T@(a[:,None]*u)/m
        C += target.T@u/m
        q += np.sum(b[:,None]*y*y)/m
        mu += a@u/m
        my += b@y/m
    return tuple(v/len(records) for v in (S,C,q,mu,my))

def fit_stats(s, rtol=1e-10):
    S,C,q,mu,my = s
    covariance = S-np.outer(mu,mu)
    covariance = (covariance+covariance.T)/2
    cross = C-np.outer(my,mu)
    values,vectors = np.linalg.eigh(covariance)
    cutoff = rtol*max(float(values[-1]),0.)
    keep = values>cutoff
    inverse = (vectors[:,keep]/values[keep])@vectors[:,keep].T
    A = cross@inverse
    b = my-A@mu
    return A,b,{'rank':int(keep.sum()),'dimension':len(mu),'eigenvalue_cutoff':cutoff}

def error(A,b,s):
    S,C,q,mu,my = s
    loss = q-2*np.sum(A*C)+np.sum((A@S)*A)+2*b@(A@mu-my)+b@b
    return float(max(loss,0.))

def fit_paired(u,y,rtol=1e-10,progress=None):
    """Exact average of individually centered pair slopes, fixed lengths.

    Only one source SVD per position is needed: sum centered future targets
    before multiplying by the common pseudoinverse. Average intercepts too.
    The average intercept is a diagnostic, not a pooled-optimal intercept.
    """
    u,y = np.asarray(u,dtype=np.float64),np.asarray(y,dtype=np.float64)
    if u.ndim!=3 or u.shape!=y.shape or len(u)<2:
        raise ValueError('Need matching [N,n,d] arrays and at least two passages.')
    mu,my = u.mean(axis=0),y.mean(axis=0)
    uc,yc = u-mu,y-my
    N,n,d = u.shape
    total,btotal,suffix,ysum = np.zeros((d,d)),np.zeros(d),np.zeros((N,d)),np.zeros(d)
    ranks=[]
    for j in range(n-1,-1,-1):
        suffix += yc[:,j]
        ysum += my[j]
        left,sv,right = np.linalg.svd(uc[:,j],full_matrices=False)
        keep = sv>np.sqrt(rtol)*sv[0]
        B = ((right[keep].T/sv[keep])@(left[:,keep].T@suffix)).T
        total += B
        btotal += ysum-B@mu[j]
        ranks.append({'position':j+17,'rank':int(keep.sum())})
        if progress: progress(n-j,n,ranks[-1])
    m=n*(n+1)/2
    return total/m,btotal/m,sorted(ranks,key=lambda r:r['position'])

def read_records(run,split,meta,length=None):
    import torch
    paths=sorted((run/split).glob('sequence_*.pt'))
    if not paths or len(paths)!=meta['splits'][split]['sequences']:
        raise ValueError(f'{split}: file count mismatch.')
    records,hashes,excluded=[],{},[]
    for p in paths:
        r=torch.load(p,map_location='cpu',weights_only=True)
        u,y=r['h_mid'].float().numpy(),r['h_final'].float().numpy()
        n=len(u)
        if (u.shape!=(n,meta['hidden_size']) or y.shape!=u.shape or len(r['input_ids'])!=n
            or r['layer']!=meta['layer'] or r['final_layer']!=meta['final_layer']
            or r['split']!=split or not np.array_equal(r['positions'].numpy(),np.arange(n))
            or not np.isfinite(u).all() or not np.isfinite(y).all()):
            raise ValueError(f'Invalid record: {p}')
        if length is not None and n<length:
            excluded.append(p.name)
            continue
        stop=length if length is not None else n
        if stop<=17: raise ValueError(f'No retained positions: {p}')
        records.append((u[16:stop-1],y[16:stop-1]))
        hashes[p.name]=sha(p)
    if not records: raise ValueError('No eligible passages.')
    return records,{'sequences':len(records),'file_sha256':hashes,'excluded_short_files':excluded}

def load_suite(fit_dir=None,run=DEFAULT_RUN,jlens_dir=None):
    if fit_dir is None:
        paths=sorted(Path(run).expanduser().glob('affine_fit_*/fit_report.json'))
        if not paths: raise FileNotFoundError('Run fit_lenses.py first.')
        fit_dir=paths[-1].parent
    fit=Path(fit_dir).expanduser().resolve()
    report=json.loads((fit/'fit_report.json').read_text())
    if report.get('format_version')!=2 or not report.get('centered'):
        raise ValueError('Expected a new affine suite fit.')
    run=Path(report['activation_run'])
    if sha(run/'metadata.json')!=report['collection_metadata_sha256']:
        raise ValueError('Collection metadata changed.')
    with np.load(fit/'lenses.npz',allow_pickle=False) as z:
        maps={name:z[name+'_A'].copy() for name in NAMES}
        means={name+'_mean':z[name+'_mean'].copy() for name in ('pooled','same')}
    jp=Path(jlens_dir).expanduser() if jlens_dir else None
    if jp is None:
        paths=sorted(run.glob('jlens_fit_*/fit_report.json'))
        if paths: jp=paths[-1].parent
    if jp:
        jr=json.loads((jp/'fit_report.json').read_text())
        for key in ('collection_metadata_sha256','source_layer','target_layer','model_name','model_revision'):
            if jr.get(key)!=report.get(key): raise ValueError('J-lens mismatch: '+key)
        with np.load(jp/'linear_lens.npz',allow_pickle=False) as z: maps['jlens']=z['A'].copy()
        report['jlens_comparison']={'directory':str(jp),'passages':jr['n_prompts'],'status':jr['status']}
    else:
        print('No cached J-lens found; evaluating regression lenses only.',flush=True)
    d=next(iter(maps.values())).shape[0]
    if any(a.shape!=(d,d) or not np.isfinite(a).all() for a in maps.values()):
        raise ValueError('Invalid saved slope.')
    if any(a.shape!=(d,) or not np.isfinite(a).all() for a in means.values()):
        raise ValueError('Invalid saved mean.')
    report['lenses_sha256']=sha(fit/'lenses.npz')
    if jp: report['jlens_sha256']=sha(jp/'linear_lens.npz')
    return fit,report,maps,means

def rank_tokens(logits,tokenizer,k):
    scores,ids=logits.float().topk(k)
    return [{'rank':r,'token_id':i,'token':tokenizer.decode([i],skip_special_tokens=False),'logit':s}
            for r,(i,s) in enumerate(zip(ids.cpu().tolist(),scores.cpu().tolist()),1)]

def check_model(model,report):
    from model_utils import MODEL_NAME
    if report['model_name']!=MODEL_NAME or report['target_layer']!=model.config.num_hidden_layers:
        raise ValueError('Model mismatch.')
    if report.get('model_revision') and report['model_revision']!=getattr(model.config,'_commit_hash',None):
        raise ValueError('Model revision mismatch.')
    if report['hidden_size']!=model.config.hidden_size: raise ValueError('Dimension mismatch.')
