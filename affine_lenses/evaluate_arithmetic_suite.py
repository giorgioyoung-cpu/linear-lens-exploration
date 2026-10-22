"""Fixed 24-problem chat arithmetic suite. No lens fitting; no plots required."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from lens_core import load_suite, layer_runs, check_model, rank_tokens, stamp


def cases():
    result=[]
    for a,b,d in [(1,2,5),(2,2,5),(2,3,3),(2,4,3),(3,4,1),(3,5,1)]:
        result.append({'family':'addition','expression':f'({a} + {b}) + {d}',
                       'targets':{'sum':a+b,'answer':a+b+d}})
    for a,b,c,d in itertools.product(range(1,4),range(1,4),range(2,5),range(1,6)):
        if (a+b)*c+d<=9:
            result.append({'family':'multiply_add','expression':f'({a} + {b}) * {c} + {d}',
                           'targets':{'sum':a+b,'product':(a+b)*c,'answer':(a+b)*c+d}})
    return result


def summarize(trials):
    summary={'n':len(trials),'correct':sum(t['correct'] for t in trials),
             'number_only':sum(t['number_only'] for t in trials),'groups':[]}
    for family in ('addition','multiply_add'):
        for group in ('all','correct','incorrect'):
            selected=[t for t in trials if t['family']==family and
                      (group=='all' or t['correct']==(group=='correct'))]
            item={'family':family,'group':group,'n':len(selected),'readouts':[]}
            if selected:
                for layer in selected[0]['readouts']:
                    for method in selected[0]['readouts'][layer]:
                        for target in selected[0]['targets']:
                            ranks=[t['readouts'][layer][method]['digit_ranks'][str(t['targets'][target])] for t in selected]
                            item['readouts'].append({'layer':int(layer),'method':method,'target':target,
                              'median_rank':float(np.median(ranks)),
                              'mean_reciprocal_rank':float(np.mean([1/r for r in ranks])),
                              'top10_fraction':float(np.mean([r<=10 for r in ranks]))})
            summary['groups'].append(item)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--jlens-checkpoint',type=int,default=118)
    p.add_argument('--max-new-tokens',type=int,default=64)
    args=p.parse_args()
    if args.max_new_tokens<1:p.error('max-new-tokens must be positive')
    import torch
    from model_utils import load_model
    from collect_activations import capture_layers
    layers=[4,8,12,16,20]
    runs=dict(layer_runs(args.run_dir))
    suites=[load_suite(run=runs[l],jlens_checkpoint=args.jlens_checkpoint) for l in layers]
    for layer,(_,report,maps,_) in zip(layers,suites):
        if report['source_layer']!=layer or 'jlens' not in maps:raise ValueError('Missing matching J-lens')
        for k in ('model_name','model_revision','target_layer','hidden_size'):
            if report[k]!=suites[0][1][k]:raise ValueError('Incompatible fits: '+k)
    model,tok,device=load_model()
    for _,report,_,_ in suites:check_model(model,report)
    digit_ids={}
    for n in range(10):
        ids=tok.encode(str(n),add_special_tokens=False)
        if len(ids)!=1 or tok.decode(ids)!=str(n):raise ValueError(f'{n} is not a complete single token')
        digit_ids[str(n)]=ids[0]
    def decode(h):
        with torch.inference_mode():
            v=torch.as_tensor(h,dtype=torch.float32,device=device)
            logits=model.lm_head(model.model.norm(v)).float().cpu()
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite readout')
        return logits
    def read(logits):
        return {'top_tokens':rank_tokens(logits,tok,10),
                'digit_ranks':{s:1+int((logits>logits[i]).sum()) for s,i in digit_ids.items()}}
    out=args.run_dir.expanduser()/('arithmetic_suite_'+stamp());out.mkdir()
    all_layers=list(range(1,len(model.model.layers)+1))
    protocol={'cases':cases(),'source_layers':layers,'saved_layers':all_layers,
              'format':'chat','do_sample':False,'max_new_tokens':args.max_new_tokens,
              'primary_position':'final formatted-prompt token; before first generated token',
              'rank':'1 + count of strictly higher logits; exact digit tokens; all digits 0-9',
              'correctness':'decoded response stripped of surrounding whitespace equals expected numeral',
              'selection':'fixed cases; retain failures; no selection using lens scores',
              'limits':'small correlated suite, including commuted operands; correctness does not prove intermediate use',
              'fits':[{'directory':str(f),'report':r} for f,r,_,_ in suites]}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    constants={l:{name:read(decode(v)) for name,v in means.items()}
               for l,(_,_,_,means) in zip(layers,suites)}
    trials=[]
    eos=model.generation_config.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    print('Results folder:',out,flush=True)
    for index,case in enumerate(cases(),1):
        instruction=f'Calculate {case["expression"]}. Answer with only the number. Do not explain your reasoning.'
        text=tok.apply_chat_template([{'role':'user','content':instruction}],tokenize=False,add_generation_prompt=True)
        ids=torch.tensor(tok.encode(text,add_special_tokens=False),dtype=torch.long)
        captured=capture_layers(model,ids,all_layers,device)
        torch.save({'input_ids':ids,'formatted_prompt':text,'layers':all_layers,
                    'h_layers':{l:captured[l] for l in all_layers},'h_final':captured['final']},out/f'case_{index:02d}_activations.pt')
        with torch.inference_mode():
            generated=model.generate(input_ids=ids.unsqueeze(0).to(device),
                attention_mask=torch.ones_like(ids).unsqueeze(0).to(device),
                max_new_tokens=args.max_new_tokens,do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        gen_ids=generated[0,len(ids):].detach().cpu().tolist()
        response=tok.decode(gen_ids,skip_special_tokens=True)
        clean=response.strip();correct=clean==str(case['targets']['answer'])
        final=read(decode(captured['final'][-1]));readouts={}
        for layer,(_,_,maps,_) in zip(layers,suites):
            h=captured[layer][-1].numpy().astype(np.float64)
            readouts[str(layer)]={name:read(decode(A@h)) for name,A in maps.items()}
            readouts[str(layer)].update(ordinary_logit_lens=read(decode(h)),
                 final_layer_readout=final,**constants[layer])
        ended=bool(gen_ids) and gen_ids[-1] in eos_ids
        trial={**case,'case_id':index,'instruction':instruction,'formatted_prompt':text,
               'input_ids':ids.tolist(),'position':len(ids),'index_base':1,
               'position_token':tok.decode([int(ids[-1])]),'response':response,'generated_ids':gen_ids,
               'correct':correct,'number_only':clean.isascii() and clean.isdigit(),
               'ended_with_eos':ended,'generation_limit_reached':len(gen_ids)>=args.max_new_tokens and not ended,
               'readouts':readouts}
        trials.append(trial)
        (out/'results.json').write_text(json.dumps({'protocol':protocol,'trials':trials,'summary':summarize(trials)},indent=2,ensure_ascii=False))
        print(f'{index}/24: {case["expression"]} = {case["targets"]["answer"]}; response {response!r}; correct={correct}',flush=True)
    summary=summarize(trials)
    lines=['# Fixed arithmetic suite','',f'Correct number-only answers: {summary["correct"]}/{len(trials)}.',
           'Readouts at final chat-template token; all cases retained.',
           'Correct answers do not establish that the intended intermediate steps were used.','',
           '| Case | Expression | Expected | Response | Correct |','|---|---|---|---|---|']
    for t in trials:lines.append(f'| {t["case_id"]} | {t["expression"]} | {t["targets"]["answer"]} | {t["response"].replace(chr(10)," ").replace("|","/")} | {t["correct"]} |')
    for group in summary['groups']:
        lines += ['',f'## {group["family"]}: {group["group"]} (n={group["n"]})','',
                  '| Layer | Method | Target | Median rank | Mean reciprocal rank | Top-10 fraction |',
                  '|---|---|---|---|---|---|']
        for r in group['readouts']:lines.append(f'| {r["layer"]} | {r["method"]} | {r["target"]} | {r["median_rank"]:.1f} | {r["mean_reciprocal_rank"]:.4f} | {r["top10_fraction"]:.3f} |')
    (out/'summary.md').write_text('\n'.join(lines))
    with (out/'digit_ranks.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','family','correct','layer','method','digit','rank'])
        for t in trials:
            for l,methods in t['readouts'].items():
                for m,r in methods.items():
                    for digit,rank in r['digit_ranks'].items():writer.writerow([t['case_id'],t['family'],t['correct'],l,m,digit,rank])
    print(f'Complete: {summary["correct"]}/24 correct. Upload results.json from {out}',flush=True)

if __name__=='__main__':main()
