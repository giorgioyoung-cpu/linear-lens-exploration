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


def main():
    import torch
    from model_utils import load_model, MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--format', choices=['chat','plain'], default='chat')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--jlens-dir', type=Path)
    parser.add_argument('--task', choices=['arithmetic','counting'], default='arithmetic')
    parser.add_argument('--prompt', default='Count to five and introspect deeply.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--jlens-checkpoint', type=int, help='Use retained J-lens prefix average of this many passages.')
    args = parser.parse_args()
    if args.max_new_tokens < 1: parser.error('max-new-tokens must be positive.')
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
