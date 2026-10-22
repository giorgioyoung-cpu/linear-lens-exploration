"""Arithmetic and multi-layer counting concept probes. Keep beside lens_core.py and model_utils.py.
Run: python evaluate_concepts.py
Optional: --format plain (default: chat), --fit-dir PATH.
Ranks only COMPLETE single-token spellings; never treats a digit prefix as a number.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from lens_core import rank_tokens, DEFAULT_RUN, load_suite, check_model

CASES = [(1,2,5), (2,2,5), (2,3,3), (2,4,3), (3,4,1), (3,5,1)]
ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = {20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}

def words(n):
    if n < 20:
        return ONES[n]
    return TENS[n//10*10] + ('-'+ONES[n%10] if n%10 else '')

def spellings(tokenizer, n):
    # Fixed alternatives, selected without looking at lens scores.
    forms = [str(n), words(n), words(n).capitalize()]
    found = {}
    for form in forms:
        for prefix in ('', ' '):
            text = prefix + form
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and tokenizer.decode(ids) == text:
                found[ids[0]] = text
    return found

# Fixed exploratory candidates, not selected from the resulting readouts.
COUNTING_CONCEPTS = ('counting', 'halfway', 'done', 'thoughts', 'AI', 'consciousness',
                     'one', 'two', 'three', 'four', 'five')


def concept_variants(tokenizer, word):
    found = {}
    for spelling in dict.fromkeys([word, word.capitalize()]):
        for prefix in ('', ' '):
            text = prefix + spelling
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and tokenizer.decode(ids) == text:
                found[ids[0]] = text
    return found


def counting_positions(prompt_length, ids, eos_ids):
    # Include the position predicting the first response token, then each
    # response token except EOS. Python zero-based internally.
    return [j for j in range(prompt_length-1, len(ids))
            if j == prompt_length-1 or int(ids[j]) not in eos_ids]


def evaluate_counting(args):
    import csv
    import torch
    from model_utils import load_model
    from collect_activations import capture_layers
    from lens_core import layer_runs, stamp
    if args.jlens_dir and (args.run_dir.expanduser()/'layers.json').exists():
        raise ValueError('For multi-layer runs, J-lens is loaded separately from each layer directory.')
    if args.fit_dir:
        suites = [load_suite(args.fit_dir, args.run_dir, args.jlens_dir, args.jlens_checkpoint)]
        base = suites[0][0]
    elif (args.run_dir.expanduser()/'layers.json').exists():
        suites = [load_suite(run=run, jlens_checkpoint=args.jlens_checkpoint) for _,run in layer_runs(args.run_dir)]
        base = args.run_dir.expanduser()
    else:
        suites = [load_suite(run=args.run_dir, jlens_dir=args.jlens_dir, jlens_checkpoint=args.jlens_checkpoint)]
        base = suites[0][0]
    layers = [report['source_layer'] for _,report,_,_ in suites]
    if len(set(layers)) != len(layers): raise ValueError('Duplicate source layer.')
    # Require matching model and target across layers.
    for _,report,_,_ in suites:
        for key in ('model_name','model_revision','target_layer','hidden_size'):
            if report[key] != suites[0][1][key]: raise ValueError('Incompatible layer fits: '+key)
    model,tok,device = load_model()
    for _,report,_,_ in suites: check_model(model,report)
    text = (tok.apply_chat_template([{'role':'user','content':args.prompt}],
             tokenize=False,add_generation_prompt=True) if args.format == 'chat' else args.prompt)
    inputs = tok(text,return_tensors='pt',add_special_tokens=False).to(device)
    n_prompt = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    ids = generated[0].detach().cpu()
    response = tok.decode(ids[n_prompt:], skip_special_tokens=True)
    print('Generated response:\n'+response, flush=True)
    # Teacher-forced replay is causal: later response tokens cannot affect
    # an earlier position. Avoid retaining full [sequence,vocabulary] logits.
    captured = capture_layers(model, ids, layers, device)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos,list) else [eos]) | {tok.eos_token_id}
    positions = counting_positions(n_prompt, ids, eos_ids)
    candidates = {w:concept_variants(tok,w) for w in COUNTING_CONCEPTS}
    out = base/('counting_'+stamp()); out.mkdir()
    torch.save({'input_ids':ids,'prompt_length':n_prompt,'layers':layers,
                'h_layers':{l:captured[l] for l in layers},'h_final':captured['final'],
                'positions':torch.arange(len(ids))}, out/'activations.pt')
    def decode(h):
        with torch.inference_mode():
            v = torch.as_tensor(h,dtype=torch.float32,device=device)
            logits = model.lm_head(model.model.norm(v)).float().cpu()
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite readout.')
        return logits
    entries, rank_rows = [], []
    lines = ['# Counting and introspection: exploratory readouts', '',
             'Prompt: '+repr(args.prompt), 'Generated response: '+repr(response), '',
             'Inspect the response above before interpreting concepts: task compliance is not assumed.',
             'Positions are one-based. A row reads the residual AFTER consuming its displayed token;',
             'the final-layer baseline predicts the NEXT token. The first row is the final prompt token.',
             'Fixed candidates use complete single-token spellings; absent variants are unavailable.',
             'Readouts alone do not establish causal use or introspection. No fitting on this prompt.', '']
    final_logits = {j:decode(captured['final'][j]) for j in positions}
    for fit,report,maps,means in suites:
        layer = report['source_layer']
        constants = {name:decode(mean) for name,mean in means.items()}
        lines += [f'## Layer {layer}', '']
        for j in positions:
            h = captured[layer][j].numpy().astype(np.float64)
            methods = {name:decode(A@h) for name,A in maps.items()}
            methods.update(ordinary_logit_lens=decode(h), final_layer_readout=final_logits[j], **constants)
            token = tok.decode([int(ids[j])], skip_special_tokens=False)
            lines += [f'### Position {j+1}: {token!r}', '']
            readouts = {}
            for name,logits in methods.items():
                top = rank_tokens(logits,tok,10)
                ranks = {}
                for concept,variants in candidates.items():
                    ranked = [{'token_id':i,'token':v,'rank':1+int((logits>logits[i]).sum())}
                              for i,v in variants.items()]
                    best = min((v['rank'] for v in ranked),default=None)
                    ranks[concept] = {'best_rank':best,'variants':ranked}
                    rank_rows.append([layer,j+1,token,name,concept,best])
                readouts[name] = {'top_tokens':top,'concepts':ranks}
                lines += [name+': '+', '.join(repr(t['token']) for t in top),
                          'Candidate ranks: '+', '.join(f'{c}={r["best_rank"]}' for c,r in ranks.items()), '']
            entries.append({'layer':layer,'position':j+1,'token':token,'readouts':readouts})
        print(f'Counting readouts complete: layer {layer}, {len(positions)} positions',flush=True)
    payload = {'task':'counting','prompt':args.prompt,'formatted_prompt':text,'response':response,
        'format':args.format,'generation':{'do_sample':False,'max_new_tokens':args.max_new_tokens,
            'generated_tokens':len(ids)-n_prompt,'ended_with_eos':int(ids[-1]) in eos_ids},
        'source_layers':layers,'index_base':1,'prompt_length':n_prompt,
        'readout':'slope only; final RMSNorm and unembedding','test_used':False,
        'fits':[{'directory':str(f),'report':r} for f,r,_,_ in suites],
        'candidate_words':list(COUNTING_CONCEPTS),'examples':entries}
    (out/'readouts.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    (out/'readouts.md').write_text('\n'.join(lines))
    with (out/'concept_ranks.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['layer','position','token','method','concept','best_rank'])
        writer.writerows(rank_rows)
    print(f'Saved counting activations and readouts: {out}',flush=True)


PAPER_OPERANDS = (1, 2, 2, 1)


def paper_arithmetic_spec():
    a, b, c, d = PAPER_OPERANDS
    return (f'calc: ( {a} + {b} ) * {c} + {d} =',
            tuple(str(n) for n in (a+b, (a+b)*c, (a+b)*c+d)))


def plot_arithmetic(entries, layers, tokens, out, targets):
    """Plot exact numeral ranks; sampled layers are discrete, not interpolated."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    methods = ['jlens', 'pooled', 'same', 'paired', 'ordinary_logit_lens']
    lookup = {(e['layer'], e['position']): e for e in entries}
    vmax = max(c['rank'] for e in entries for m in methods
               for c in e['readouts'][m]['concepts'].values() if c['rank'] is not None)
    for method in methods:
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(tokens)*0.45), 8), constrained_layout=True)
        for ax, number in zip(axes, targets):
            values = np.array([[lookup[l,j+1]['readouts'][method]['concepts'][number]['rank']
                                for j in range(len(tokens))] for l in layers], dtype=float)
            im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', interpolation='nearest',
                           norm=LogNorm(vmin=1, vmax=max(10,vmax)), cmap='viridis_r')
            ax.set_yticks(range(len(layers)), layers)
            ax.set_xticks(range(len(tokens)), [f'{j+1}: {t!r}' for j,t in enumerate(tokens)], rotation=60, ha='right')
            ax.set_ylabel('Block'); ax.set_title(f'{method}: numeral {number}')
        fig.colorbar(im, ax=axes, label='Vocabulary rank (log scale; lower is better)')
        fig.savefig(out/f'heatmap_{method}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax, number in zip(axes, targets):
        for method in methods:
            ranks = [lookup[l,len(tokens)]['readouts'][method]['concepts'][number]['rank'] for l in layers]
            ax.plot(layers, [np.nan if r is None else r for r in ranks], marker='o', label=method)
        ax.set_yscale('log'); ax.invert_yaxis(); ax.set_xticks(layers)
        ax.set_xlabel('Block (lines connect sampled layers only)'); ax.set_title(f'Final prompt token: {number}')
    axes[0].set_ylabel('Vocabulary rank (lower is better)')
    axes[-1].legend(fontsize=7)
    fig.savefig(out/'final_position_ranks.png',dpi=180); plt.close(fig)


def evaluate_paper_arithmetic(args):
    """Figure-17-inspired prompt probe; no arithmetic-specific fitting."""
    import csv
    import torch
    from model_utils import load_model
    from collect_activations import capture_layers
    from lens_core import layer_runs, stamp
    if args.format != 'plain':
        raise ValueError('Use --format plain for the exact calc prompt; no chat wrapper is added.')
    if args.jlens_checkpoint is None:
        raise ValueError('Specify --jlens-checkpoint explicitly (our agreed run uses 118).')
    if args.fit_dir or args.jlens_dir:
        raise ValueError('Use the multi-layer --run-dir for this five-layer probe.')
    runs = dict(layer_runs(args.run_dir))
    layers = [4,8,12,16,20]
    suites = [load_suite(run=runs[l], jlens_checkpoint=args.jlens_checkpoint) for l in layers]
    for layer, (_,report,maps,_) in zip(layers,suites):
        if report['source_layer'] != layer or 'jlens' not in maps:
            raise ValueError('Missing or mismatched lens at layer '+str(layer))
        for key in ('model_name','model_revision','target_layer','hidden_size'):
            if report[key] != suites[0][1][key]: raise ValueError('Incompatible fits: '+key)
    model,tok,device = load_model()
    for _,report,_,_ in suites: check_model(model,report)
    prompt, targets = paper_arithmetic_spec()
    ids = torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)
    tokens = [tok.decode([int(i)],skip_special_tokens=False) for i in ids]
    # Exact decimal tokens, not their first digit or a best-of-variants score.
    candidates = {}
    for number in targets:
        encoded = tok.encode(number,add_special_tokens=False)
        candidates[number] = {'encoding':encoded, 'token_id':encoded[0]
                              if len(encoded)==1 and tok.decode(encoded)==number else None}
    print('Numeral tokenization:', candidates, flush=True)
    unavailable = [n for n,c in candidates.items() if c['token_id'] is None]
    if unavailable:
        raise ValueError('Cannot rank these targets as complete single tokens: '+
                         ', '.join(unavailable)+'. No arithmetic activations or readouts were computed.')
    all_layers = list(range(1,len(model.model.layers)+1))
    captured = capture_layers(model,ids,all_layers,device)
    out = args.run_dir.expanduser()/('arithmetic_paper_'+stamp()); out.mkdir()
    torch.save({'input_ids':ids,'prompt':prompt,'index_base':1,'layers':all_layers,
                'h_layers':{l:captured[l] for l in all_layers},'h_final':captured['final']},out/'activations.pt')
    with torch.inference_mode():
        generated = model.generate(input_ids=ids.unsqueeze(0).to(device),
            attention_mask=torch.ones_like(ids).unsqueeze(0).to(device),
            max_new_tokens=args.max_new_tokens,do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    continuation_ids = generated[0,len(ids):].detach().cpu().tolist()
    response = tok.decode(continuation_ids,skip_special_tokens=True)
    print('Generated continuation:',repr(response),flush=True)
    def decode(h):
        with torch.inference_mode():
            v=torch.as_tensor(h,dtype=torch.float32,device=device)
            logits=model.lm_head(model.model.norm(v)).float().cpu()
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite readout.')
        return logits
    finals = [decode(h) for h in captured['final']]
    entries, rows = [], []
    lines=['# Five-layer arithmetic probe','', 'Prompt: '+repr(prompt),
           'Generated continuation: '+repr(response),'',
           'Exact decimal-token ranks; unavailable multi-token numerals have null ranks.',
           'Positions are one-based, after consuming the displayed token. No generated-answer activations are inspected.',
           'All affine readouts discard intercepts. Readouts do not establish causal use.',
           'Adaptation of Figure 17: operands changed to keep all targets single-token (3, 6, 7).',
           'Qwen, five sampled layers, existing WikiText lens fits; not an exact replication.','']
    for fit,report,maps,means in suites:
        layer=report['source_layer']
        constants={name:decode(mean) for name,mean in means.items()}
        for j,token in enumerate(tokens):
            h=captured[layer][j].numpy().astype(np.float64)
            methods={name:decode(A@h) for name,A in maps.items()}
            methods.update(ordinary_logit_lens=decode(h),final_layer_readout=finals[j],**constants)
            readouts={}
            lines += [f'## Layer {layer}, position {j+1}: {token!r}','']
            for name,logits in methods.items():
                concepts={}
                for number,candidate in candidates.items():
                    i=candidate['token_id']
                    rank=None if i is None else 1+int((logits>logits[i]).sum())
                    concepts[number]={'token_id':i,'rank':rank}
                    rows.append([layer,j+1,token,name,number,i,rank])
                top=rank_tokens(logits,tok,10)
                readouts[name]={'top_tokens':top,'concepts':concepts}
                lines += [name+': '+', '.join(repr(t['token']) for t in top),
                          'Ranks: '+', '.join(f'{n}={c["rank"]}' for n,c in concepts.items()),'']
            entries.append({'layer':layer,'position':j+1,'token':token,'readouts':readouts})
        print(f'Arithmetic readouts complete: layer {layer}',flush=True)
    payload={'task':'arithmetic_paper','prompt':prompt,'format':'plain','input_ids':ids.tolist(),
             'tokens':tokens,'index_base':1,'response':response,'generated_ids':continuation_ids,
             'max_new_tokens':args.max_new_tokens,'do_sample':False,'expected_answer':targets[-1],
             'operands':list(PAPER_OPERANDS),'targets':list(targets),
             'original_paper_prompt':'calc: ( 4 + 17 ) * 2 + 7 =',
             'adaptation':'Single-digit targets to allow exact single-token vocabulary ranks.',
             'exact_answer_match':response.strip()==targets[-1],'candidate_tokenization':candidates,
             'evaluated_layers':layers,'saved_activation_layers':all_layers,
             'rank_convention':'1 + count of strictly higher logits; exact decimal token only',
             'fits':[{'directory':str(f),'report':r} for f,r,_,_ in suites],
             'reference':'https://www.transformer-circuits.pub/2026/workspace/index.html',
             'examples':entries}
    (out/'readouts.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    (out/'readouts.md').write_text('\n'.join(lines))
    with (out/'concept_ranks.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','position','token','method','numeral','token_id','rank']);writer.writerows(rows)
    plot_arithmetic(entries,layers,tokens,out,targets)
    print(f'Saved arithmetic results: {out}',flush=True)


def main():
    import torch
    from model_utils import load_model, MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--format', choices=['chat','plain'], default='chat')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--jlens-dir', type=Path)
    parser.add_argument('--task', choices=['arithmetic','counting','arithmetic-paper'], default='arithmetic')
    parser.add_argument('--prompt', default='Count to five and introspect deeply.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--jlens-checkpoint', type=int, help='Use retained J-lens prefix average of this many passages.')
    args = parser.parse_args()
    if args.max_new_tokens < 1: parser.error('max-new-tokens must be positive.')
    if args.task == 'arithmetic-paper':
        evaluate_paper_arithmetic(args)
        return
    if args.task == 'counting':
        evaluate_counting(args)
        return
    if (args.run_dir.expanduser()/'layers.json').exists() and not args.fit_dir:
        from lens_core import dispatch_layers
        dispatch_layers(args.run_dir)
        return
    fit, report, maps, means = load_suite(args.fit_dir, args.run_dir, args.jlens_dir, args.jlens_checkpoint)
    print(f'Using fit: {fit}', flush=True)
    model, tok, device = load_model()
    check_model(model, report)
    layer = report['source_layer']

    def decode(h):
        with torch.inference_mode():
            v = torch.as_tensor(h, dtype=torch.float32, device=device)
            logits = model.lm_head(model.model.norm(v)).float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite readout.')
        return logits

    for digit in range(10):
        candidates = spellings(tok,digit)
        if str(digit) not in candidates.values():
            raise ValueError(f'Digit {digit} is not a complete single token; stop this probe.')
    constants = {name:decode(mean) for name,mean in means.items()}
    examples = []
    lines = ['# Arithmetic intermediate-concept probe', '',
             f'Layer {layer}; {args.format} formatting. Six fixed exploratory problems.',
             'Readout at the final prompt token before generation. No fitting or test-set access.',
             'Ranks use complete single-token numeral/English-word spellings only. Lower is better.',
             'Missing spellings are unavailable, not a failed conceptual readout. Digit prefixes are excluded.',
             'A low rank is evidence of decodability, not proof that the model uses that intermediate.',
             'One layer cannot establish the order of computation across layers.',
             'All digits 0–9 are scored on every problem; compare the same numeral across relevant and irrelevant contexts.',
             'Correct final answers do not establish that the model used the parenthesized intermediate.', '']
    for a,b,e in CASES:
        expression = f'({a} + {b}) + {e}'
        values = {'intermediate':a+b, 'answer':a+b+e}
        # Record the same candidate set on every problem, including all incorrect digits.
        controls = {f'digit_{n}': n for n in range(10)}
        targets = {**values, **controls}
        prompt = f'Calculate {expression}. Answer with only the number.'
        text = (tok.apply_chat_template([{'role':'user','content':prompt}],
                tokenize=False,add_generation_prompt=True) if args.format=='chat'
                else expression+' =')
        inputs = tok(text,return_tensors='pt',add_special_tokens=False).to(device)
        captured = {}
        def capture(module, inp, out):
            h = out[0] if isinstance(out,tuple) else out
            captured['h'] = h[0,-1].detach().float().cpu().numpy().copy()
        hook = model.model.layers[layer-1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                result = model(**inputs,use_cache=False,return_dict=True)
                final = result.logits[0,-1].float().cpu()
                del result
        finally:
            hook.remove()
        h = captured['h']
        methods = {name:decode(A@h.astype(np.float64)) for name,A in maps.items()}
        methods.update(ordinary_logit_lens=decode(h), final_layer_readout=final, **constants)
        readouts = {}
        print(f'\n{expression} = {values["answer"]}; intermediate {a+b}',flush=True)
        lines += ['## '+expression, '', 'Expected values: '+json.dumps(values), '']
        for name, logits in methods.items():
            top = rank_tokens(logits,tok,10)
            ranks = {}
            for label,n in targets.items():
                entries = [{'token_id':i,'token':t,'rank':1+int((logits>logits[i]).sum().item())}
                           for i,t in spellings(tok,n).items()]
                ranks[label] = {'value':n,'variants':entries,
                                'best_rank':min((v['rank'] for v in entries), default=None),
                                'numeral_rank':next((v['rank'] for v in entries if v['token']==str(n)),None)}
            readouts[name] = {'top_tokens':top,'concepts':ranks}
            brief = ', '.join(f'{label}={v["value"]}: rank {v["best_rank"]}' for label,v in ranks.items() if label in values)
            print(name+': '+brief,flush=True)
            lines += ['### '+name, '', ', '.join(repr(t['token']) for t in top), '',
                      '| Target | Value | Numeral rank | Best variant rank | Single-token variants and ranks |',
                      '|---|---:|---:|---:|---|']
            for label,v in ranks.items():
                variants = '; '.join(f'{x["token"]!r}: {x["rank"]}' for x in v['variants']) or 'Unavailable'
                lines.append(f'| {label} | {v["value"]} | {v["numeral_rank"]} | {v["best_rank"]} | {variants} |')
            lines.append('')
        # Hook removed: generated tokens cannot overwrite the inspected activation.
        with torch.inference_mode():
            generated = model.generate(**inputs,max_new_tokens=16,do_sample=False,
                                       pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        answer = tok.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        print('Generated answer: '+repr(answer),flush=True)
        lines += ['Generated continuation: '+repr(answer),
                  'Exact answer match: '+str(answer.strip()==str(values['answer'])), '']
        examples.append({'expression':expression,'formatted_prompt':text,'expected':values,
                         'readout_position':inputs['input_ids'].shape[1], 'index_base':1,
                         'generated_answer':answer,'exact_answer_match':answer.strip()==str(values['answer']),
                         'readouts':readouts})
    correct = sum(x['exact_answer_match'] for x in examples)
    print(f'\nExact final answers: {correct}/{len(examples)}',flush=True)
    lines += ['## Answer accuracy', '', f'{correct}/{len(examples)} exact final answers.',
              'Retain all cases; distinguish correct-answer cases when interpreting intermediates.', '']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = fit/f'arithmetic_simple_{args.format}_{stamp}'
    out.mkdir(exist_ok=False)
    (out/'readouts.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'readouts.json').write_text(json.dumps({'fit_directory':str(fit),'source_layer':layer,
        'format':args.format,'provenance':report, 'readout':'slope only; no intercept',
        'rank_convention':'1 + number of strictly higher logits; best over listed single-token variants',
        'test_used':False,'examples':examples},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nSaved arithmetic results to: {out}',flush=True)

if __name__ == '__main__':
    main()
