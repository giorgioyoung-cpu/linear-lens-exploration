"""Layer-12 arithmetic concept probe. Keep beside lens_core.py and model_utils.py.
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

def main():
    import torch
    from model_utils import load_model, MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--format', choices=['chat','plain'], default='chat')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--jlens-dir', type=Path)
    args = parser.parse_args()
    fit, report, maps, means = load_suite(args.fit_dir, args.run_dir, args.jlens_dir)
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
