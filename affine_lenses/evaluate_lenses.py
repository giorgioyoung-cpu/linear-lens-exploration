"""Compare all affine slopes and cached J-lens on WikiText validation residuals."""
import argparse
import json
from pathlib import Path
import numpy as np
from lens_core import DEFAULT_RUN,load_suite,rank_tokens,stamp,sha,check_model

def main():
    import torch
    from model_utils import load_model
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-dir',type=Path)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--jlens-dir',type=Path)
    p.add_argument('--positions',choices=['retained','legacy'],default='retained')
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--print-sequences',type=int,default=3)
    args=p.parse_args()
    if args.top_k<1 or args.print_sequences<0: p.error('Invalid display counts.')
    fit,report,maps,means=load_suite(args.fit_dir,args.run_dir,args.jlens_dir)
    model,tok,device=load_model(); check_model(model,report)
    def decode(h):
        with torch.inference_mode():
            logits=model.lm_head(model.model.norm(torch.as_tensor(h,dtype=torch.float32,device=device)))
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite readout.')
        return rank_tokens(logits,tok,min(args.top_k,len(logits)))
    constants={name:decode(v) for name,v in means.items()}
    run=Path(report['activation_run']); examples=[]
    hashes=report['data']['validation']['file_sha256']
    if set(hashes)!={p.name for p in (run/'validation').glob('sequence_*.pt')}:
        raise ValueError('Validation file set changed.')
    for k,(filename,digest) in enumerate(sorted(hashes.items())):
        path=run/'validation'/filename
        if sha(path)!=digest: raise ValueError('Validation data changed: '+filename)
        r=torch.load(path,map_location='cpu',weights_only=True)
        u,y=r['h_mid'].double().numpy(),r['h_final'].double().numpy()
        ids=r['input_ids'].tolist(); n=len(ids)
        if u.shape!=(n,report['hidden_size']) or y.shape!=u.shape or r['layer']!=report['source_layer']:
            raise ValueError('Invalid residual record.')
        positions=sorted({n//2,n-1}) if args.positions=='legacy' else sorted({16+(n-18)//2,n-2})
        for j in positions:
            methods={name:decode(A@u[j]) for name,A in maps.items()}
            methods.update(ordinary_logit_lens=decode(u[j]),final_layer_readout=decode(y[j]),**constants)
            context=tok.decode(ids[max(0,j-23):j+1],skip_special_tokens=False)
            examples.append({'sequence_file':filename,'position':j+1,'index_base':1,'sequence_length':n,
                'current_token':tok.decode([ids[j]]),'context_last_24_tokens':context,
                'full_prefix':tok.decode(ids[:j+1],skip_special_tokens=False),'readouts':methods})
            if k<args.print_sequences:
                print(f'\n{filename}, position {j+1}/{n}\nContext: {context!r}',flush=True)
                for name,tokens in methods.items(): print(name+': '+', '.join(repr(t['token']) for t in tokens),flush=True)
    summary={}
    for name in examples[0]['readouts']:
        if name=='final_layer_readout': continue
        overlaps=[];hits=agreements=0
        for e in examples:
            pred=[t['token_id'] for t in e['readouts'][name]]
            target=[t['token_id'] for t in e['readouts']['final_layer_readout']]
            overlaps.append(len(set(pred)&set(target)));hits+=int(target[0] in pred);agreements+=int(pred[0]==target[0])
        summary[name]={'mean_top_k_overlap':float(np.mean(overlaps)),'final_top1_in_top_k':hits,
                       'top1_agreement':agreements,'positions':len(examples)}
    out=fit/('wikitext_'+args.positions+'_'+stamp());out.mkdir()
    output={'fit_directory':str(fit),'provenance':report,'positions':args.positions,'index_base':1,
            'test_used':False,'readout':'final RMSNorm and unembedding of A h; no intercept',
            'summary':summary,'examples':examples}
    (out/'readouts.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    lines=['# WikiText affine slope readouts','','Validation only; one-based positions. All regression readouts discard the intercept.',
           'Output agreement is a secondary diagnostic, not evidence of intermediate concepts.',
           'J-lens (if present) retains its original sample count and sum-target scaling.','']
    for e in examples:
        lines += [f'## {e["sequence_file"]}, position {e["position"]}/{e["sequence_length"]}','','```text',e['context_last_24_tokens'],'```','']
        for name,tokens in e['readouts'].items(): lines += ['**'+name+'**','','```text',', '.join(repr(t['token']) for t in tokens),'```','']
    lines += ['## Agreement with final-layer readout','','```json',json.dumps(summary,indent=2),'```','']
    (out/'readouts.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2));print(f'Saved: {out}')

if __name__=='__main__': main()
