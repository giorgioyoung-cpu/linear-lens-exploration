"""Benchmark or estimate a layer-12 Jacobian lens using cached training token IDs.

Default: a small benchmark, NOT a full lens. Run --fit --prompts 8 for a pilot.
Estimator matches the positional reductions in anthropics/jacobian-lens fitting.py:
sum valid target positions, mean valid source positions, mean passages.
Independent Qwen adapter; no attention freezing; parameters are not trained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np

DEFAULT_RUN=Path.home()/'jspace-data/activations/layer12_20261014T191345_397425Z'
REFERENCE='https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py'


def synchronize(device):
    import torch
    if device.type=='mps': torch.mps.synchronize()
    elif device.type=='cuda': torch.cuda.synchronize(device)


def valid_slice(n,skip=16):
    if skip<0 or n<=skip+1: raise ValueError('Sequence has no valid positions.')
    return slice(skip,n-1)


def reduce_gradient_rows(final, source, row_ids, skip=16, retain_graph=True):
    """Batch element b encodes output coordinate row_ids[b]; rows are output x input."""
    import torch
    valid=valid_slice(final.shape[1],skip)
    cotangent=torch.zeros_like(final)
    for b,row in enumerate(row_ids): cotangent[b,valid,row]=1
    gradient=torch.autograd.grad(final,source,grad_outputs=cotangent,
                                 retain_graph=retain_graph,create_graph=False)[0]
    return gradient[:len(row_ids),valid,:].mean(dim=1).detach()


def self_test():
    """Check causal aggregation, orientation, and masks on a known linear sequence map."""
    import torch
    g=torch.Generator().manual_seed(4)
    n,d,skip=6,3,1
    blocks=torch.randn(n,n,d,d,generator=g,dtype=torch.float64)
    for i in range(n): blocks[i,i+1:]=0
    source=torch.randn(d,n,d,generator=g,dtype=torch.float64,requires_grad=True)
    final=torch.einsum('ijok,bjk->bio',blocks,source)
    got=reduce_gradient_rows(final,source,list(range(d)),skip,False)
    expected=blocks[skip:n-1,skip:n-1].sum(dim=(0,1))/(n-1-skip)
    torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-10)


def forward_from_source(model, ids, layer, batch, perturb=None, track=True):
    """Run unchanged prefix, replace chosen block output with an independent leaf.

    Only later blocks enter the graph. Output target is BEFORE final RMSNorm.
    perturb is a whole [n,d] residual perturbation for the finite-difference check.
    """
    import torch
    captured={}
    def source_hook(module, args, output):
        h=output[0] if isinstance(output,tuple) else output
        h=h.detach().clone()
        if perturb is not None: h=h+perturb.unsqueeze(0)
        h.requires_grad_(track)
        captured['source']=h
        return (h,)+output[1:] if isinstance(output,tuple) else h
    def target_hook(module,args): captured['target']=args[0]
    h1=model.model.layers[layer-1].register_forward_hook(source_hook)
    h2=model.model.norm.register_forward_pre_hook(target_hook)
    try:
        with torch.set_grad_enabled(track):
            result=model.model(input_ids=ids.unsqueeze(0).repeat(batch,1),
                               use_cache=False,return_dict=True)
        del result
    finally:
        h1.remove();h2.remove()
    return captured['source'],captured['target']


def estimate_rows(model,ids,layer,dim_batch,row_ids,cached,skip=16,progress=True):
    import torch
    device=ids.device
    synchronize(device);start=perf_counter()
    source,final=forward_from_source(model,ids,layer,dim_batch)
    synchronize(device);forward_seconds=perf_counter()-start
    n,d=final.shape[1:]
    valid_slice(n,skip)
    checks={}
    for label,got,key in [('source',source,'h_mid'),('target',final,'h_final')]:
        actual=got[0].detach().float().cpu()
        expected=cached[key].float()
        relative=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
        checks[label+'_relative_difference_from_cache']=relative
        if relative>1e-3 or not torch.isfinite(actual).all():
            raise ValueError(f'{label} differs from cached collection: relative error {relative}.')
    out=np.empty((len(row_ids),d),dtype=np.float64)
    times=[]
    for k,start_row in enumerate(range(0,len(row_ids),dim_batch)):
        rows=row_ids[start_row:start_row+dim_batch]
        synchronize(device);t=perf_counter()
        block=reduce_gradient_rows(final,source,rows,skip,
                                   retain_graph=start_row+dim_batch<len(row_ids))
        out[start_row:start_row+len(rows)]=block.float().cpu().numpy()
        synchronize(device);times.append(perf_counter()-t)
        if not np.isfinite(out[start_row:start_row+len(rows)]).all():
            raise ValueError('Nonfinite Jacobian row.')
        if progress and (k==0 or (k+1)%16==0 or start_row+dim_batch>=len(row_ids)):
            print(f'  Jacobian rows: {min(start_row+dim_batch,len(row_ids))}/{len(row_ids)}',flush=True)
    del source,final
    return out,{'forward_seconds':forward_seconds,'backward_batch_seconds':times,
                'elapsed_seconds':perf_counter()-start,**checks}


def finite_difference_check(model,ids,layer,row,row_vector,skip=16):
    import torch
    n=len(ids);d=len(row_vector);valid=valid_slice(n,skip)
    direction=np.random.default_rng(91).normal(size=d)
    direction/=np.linalg.norm(direction)
    perturb=torch.zeros(n,d,device=ids.device,dtype=torch.float32)
    perturb[valid]=torch.as_tensor(direction,device=ids.device,dtype=torch.float32)
    expected=float(row_vector@direction)
    tests=[]
    for eps in (0.03,0.01):
        values=[]
        for sign in (1,-1):
            source,final=forward_from_source(model,ids,layer,1,sign*eps*perturb,False)
            values.append(float(final[0,valid,row].float().cpu().double().mean()))
            del source,final
        observed=(values[0]-values[1])/(2*eps)
        relative=abs(observed-expected)/max(abs(observed),abs(expected),1e-8)
        tests.append({'epsilon':eps,'finite_difference':observed,'autograd':expected,
                      'relative_difference':relative,
                      'passed':abs(observed-expected)<=0.01+0.1*abs(expected)})
    return {'checks':tests,'passed':any(t['passed'] for t in tests),
            'note':'float32 directional numerical sanity check, not a convergence guarantee'}


def atomic_npz(path,**arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez(f,**arrays)
    temp.replace(path)


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from model_utils import MODEL_NAME,DATA_DIR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',action='store_true')
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--prompts',type=int,default=8)
    p.add_argument('--dim-batch',type=int,default=2)
    p.add_argument('--benchmark-rows',type=int,default=8)
    p.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='auto')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if min(args.prompts,args.dim_batch,args.benchmark_rows)<1: p.error('Counts must be positive.')
    if args.resume and not args.fit: p.error('--resume requires --fit.')
    self_test();print('Synthetic Jacobian aggregation check passed.',flush=True)
    run=args.run_dir.expanduser().resolve()
    meta_path=run/'metadata.json';meta=json.loads(meta_path.read_text())
    if meta.get('status')!='complete' or meta['model_name']!=MODEL_NAME:
        raise ValueError('Expected our completed Qwen activation collection.')
    layer=meta['layer'];d=meta['hidden_size']
    files=sorted((run/'train').glob('sequence_*.pt'))
    if len(files)!=meta['splits']['train']['sequences']:raise ValueError('Training file count mismatch.')
    eligible=[]
    # Use the same full-length eligible population as the pair-average pilot.
    for file in files:
        raw=file.read_bytes();record=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if record['split']!='train' or record['layer']!=layer:raise ValueError('Invalid record.')
        if len(record['input_ids'])==128:
            eligible.append({'filename':file.name,'sha256':hashlib.sha256(raw).hexdigest()})
    if args.prompts>len(eligible):p.error(f'Only {len(eligible)} full-length training passages available.')
    chosen=eligible[:args.prompts]
    device_name=args.device
    if device_name=='auto': device_name='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device(device_name)
    print(f'Loading cached model on {device}, float32, eager attention...',flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=str(DATA_DIR/'models'),
        local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').to(device)
    model.eval();model.requires_grad_(False)
    if model.config.hidden_size!=d or model.config.num_hidden_layers!=meta['final_layer']:
        raise ValueError('Model dimensions mismatch.')
    if meta.get('model_revision') and meta['model_revision']!=getattr(model.config,'_commit_hash',None):
        raise ValueError('Model revision mismatch.')
    config={'model_name':MODEL_NAME,'model_revision':meta.get('model_revision'),'activation_run':str(run),
        'collection_metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        'source_layer':layer,'target_layer':meta['final_layer'],'dimension':d,'skip_first':16,
        'position_rule':'source and target indices 16 through n-2 inclusive',
        'estimator':'mean passages; mean valid sources; SUM valid targets (causal)',
        'attention':'eager, differentiated normally','dtype':'float32 autograd; float64 accumulation',
        'selected_training_files':chosen,'reference':REFERENCE,'device':str(device),
        'dim_batch':args.dim_batch,'test_evaluated':False}
    def read(entry):
        raw=(run/'train'/entry['filename']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Training file changed.')
        r=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        return r,r['input_ids'].to(device)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not args.fit:
        record,ids=read(chosen[0])
        row_ids=list(range(min(args.benchmark_rows,d)))
        J,timing=estimate_rows(model,ids,layer,args.dim_batch,row_ids,record)
        check=finite_difference_check(model,ids,layer,row_ids[0],J[0])
        batch_times=timing['backward_batch_seconds']
        per_batch=float(np.median(batch_times[1:] or batch_times))
        estimate=timing['forward_seconds']+math.ceil(d/args.dim_batch)*per_batch
        output={**config,'mode':'benchmark only; incomplete rows cannot be used as a lens',
                'rows':row_ids,'timing':timing,'finite_difference':check,
                'estimated_seconds_per_full_prompt':estimate,
                'estimated_seconds_for_requested_prompts':estimate*args.prompts,
                'packages':{n:version(n) for n in ['torch','transformers','numpy']}}
        out=run/f'jlens_benchmark_{stamp}';out.mkdir()
        (out/'benchmark.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        print('Finite-difference check: '+('PASS' if check['passed'] else 'FAIL — do not fit yet'),flush=True)
        print(json.dumps(check,indent=2),flush=True)
        print(f'Estimated full 896-row prompt time: {estimate/60:.2f} minutes',flush=True)
        print(f'Estimated {args.prompts}-prompt fit: {estimate*args.prompts/60:.2f} minutes (rough extrapolation)',flush=True)
        print(f'Benchmark saved to: {out}',flush=True)
        return

    if args.resume:
        out=args.resume.expanduser().resolve()
        old=json.loads((out/'config.json').read_text())
        if old!=config:raise ValueError('Resume arguments/data differ from original fit configuration.')
        with np.load(out/'checkpoint.npz',allow_pickle=False) as saved:
            total=saved['sum_J'].copy();done=int(saved['done']);mean_total=saved['mean_total'].copy()
        print(f'Resuming after {done}/{len(chosen)} complete passages.',flush=True)
    else:
        out=run/f'jlens_fit_{stamp}';out.mkdir(exist_ok=False)
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        total=np.zeros((d,d));mean_total=np.zeros(d);done=0
        atomic_npz(out/'checkpoint.npz',sum_J=total,done=np.array(done),mean_total=mean_total)
    print(f'Fit directory (for resume): {out}',flush=True)
    for entry in chosen[done:]:
        print(f'Passage {done+1}/{len(chosen)}: {entry["filename"]}',flush=True)
        record,ids=read(entry)
        J,timing=estimate_rows(model,ids,layer,args.dim_batch,list(range(d)),record)
        if done==0:
            check=finite_difference_check(model,ids,layer,0,J[0])
            if not check['passed']:raise ValueError(f'Finite-difference check failed: {check}')
        total+=J;mean_total+=record['h_final'][16:-1].double().mean(dim=0).numpy();done+=1
        atomic_npz(out/'checkpoint.npz',sum_J=total,done=np.array(done),mean_total=mean_total)
        # Only COMPLETE passage estimates enter the readable lens.
        atomic_npz(out/'linear_lens.npz',A=total/done)
        means=out/'mean_comparison_jlens';means.mkdir(exist_ok=True)
        np.save(means/'training_target_mean.npy',mean_total/done)
        report={**config,'format_version':1,'n_prompts':done,'requested_prompts':len(chosen),
            'status':'complete' if done==len(chosen) else 'partial',
            'intercept':False,'centered':False,'pair_set':'16 <= j <= i < n-1',
            'last_prompt_seconds':timing['elapsed_seconds'],
            'baseline_mean':'uniform valid target positions, then uniform completed training passages'}
        temp=out/'fit_report.json.tmp';temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(out/'fit_report.json')
        print(f'Saved {done}-passage estimate; last passage {timing["elapsed_seconds"]:.1f}s.',flush=True)
    print(f'\nSaved J-lens: {out}',flush=True)


if __name__=='__main__': main()
