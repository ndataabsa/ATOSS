import os
import re
import numpy as np
import pickle
from functools import partial
import time
from tqdm import tqdm
from collections import Counter
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from mvp.data_utils import *

def extract_spans_paraphrase(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(' because ')
                c = opinion2word.get(c[6:], 'nope')    # 'good' -> 'positive'
                a, b = ab.split(' is ')
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                a, b, c = '', '', ''
            quads.append((a, b, c))
    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                ac_sp, at_sp = s.split(' because ')
                
                ac, sp = ac_sp.split(' is ')
                at, sp2 = at_sp.split(' is ')
                
                sp = opinion2word.get(sp, 'nope')
                sp2 = opinion2word.get(sp2, 'nope')
                if sp != sp2:
                    print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                ac, at, sp = '', '', ''
            
            quads.append((ac, at, sp))
    elif task in ["asqp", "acos"]:
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads

def extract_spans_para(seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'null'
        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))

    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    if verbose:
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, paraphrase, verbose=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        if paraphrase:
            gold_list = extract_spans_paraphrase('asqp', gold_seqs[i], 'gold')
            pred_list = extract_spans_paraphrase('asqp', pred_seqs[i], 'pred')
        else:
            gold_list = extract_spans_para(gold_seqs[i], 'gold')
            pred_list = extract_spans_para(pred_seqs[i], 'pred')
        if verbose and i < 10:

            print("gold ", gold_seqs[i])
            print("pred ", pred_seqs[i])
            print()

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds

def evaluate(args, model, task, data, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    tasks, datas, sents, _ = read_line_examples_from_file(
        f'../data/{task}/{data}/{data_type}.txt', task, data, lowercase=False)

    outputs, targets, probs = [], [], []
    num_path = args.num_path
    if task in ['aste', 'tasd']:
        num_path = min(5, num_path)

    cache_file = os.path.join(
        args.output_dir, "result_{}{}{}_{}_{}_path{}_beam{}.pickle".format(
            "best_" if args.load_ckpt_name else "",
            "cd_" if args.constrained_decode else "", task, data, data_type, num_path,
            args.beam_size))
    if args.load_path_cache:
        with open(cache_file, 'rb') as handle:
            (outputs, targets, probs) = pickle.load(handle)
    else:
        dataset = ABSADataset(model.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              top_k=num_path,
                              args=args,
                              max_len=args.max_seq_length)
        data_loader = DataLoader(dataset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=2)
        device = torch.device('cuda:0')
        model.model.to(device)
        model.model.eval()

        for batch in tqdm(data_loader):
            # beam search
            outs = model.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=args.beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )

            dec = [
                model.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]
            target = [
                model.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            outputs.extend(dec)
            targets.extend(target)

        # save outputs and targets
        with open(cache_file, 'wb') as handle:
            pickle.dump((outputs, targets, probs), handle)

    if args.multi_path:
        targets = targets[::num_path]

        # get outputs
        _outputs = outputs # backup
        outputs = [] # new outputs
        if args.agg_strategy == 'post_rank':
            inputs = [ele for ele in sents for _ in range(num_path)]
            assert len(_outputs) == len(inputs), (len(_outputs), len(inputs))
            preds = [[o] for o in _outputs] 
            model_path = os.path.join(args.output_dir, "final")
            scores = cal_entropy(inputs, preds, model_path, model.tokenizer)

        for i in range(0, len(targets)):
            o_idx = i * num_path
            multi_outputs = _outputs[o_idx:o_idx + num_path]

            if args.agg_strategy == 'post_rank':
                multi_probs = scores[o_idx:o_idx + args.num_path]
                assert len(multi_outputs) == len(multi_probs)

                sorted_outputs = [i for _,i in sorted(zip(multi_probs,multi_outputs))]
                outputs.append(sorted_outputs[0])
                continue
            elif args.agg_strategy == "pre_rank":
                outputs.append(multi_outputs[0])
                continue
            elif args.agg_strategy == 'rand':
                outputs.append(random.choice(multi_outputs))
                continue
            elif args.agg_strategy == 'heuristic':
                # aspect term > opinion term = aspect category > sentiment polarity
                optim_orders_all = get_orders_all()
                heuristic_orders =  get_orders_heuristic()
                index = optim_orders_all[task][data].index(heuristic_orders[task][0])
                outputs.append(multi_outputs[index])
                # at, ot/ac, sp
                continue
            elif args.agg_strategy == 'vote':
                all_quads = []
                for s in multi_outputs:
                    all_quads.extend(
                        extract_spans_para(seq=s, seq_type='pred'))

                output_quads = []
                counter = dict(Counter(all_quads))
                for quad, count in counter.items():
                    # keep freq >= num_path / 2
                    if count >= len(multi_outputs) / 2:
                        output_quads.append(quad)

                # recover output
                output = []
                for q in output_quads:
                    ac, at, sp, ot = q
                    if tasks[i] == "aste":
                        if 'null' not in [at, ot, sp]:  # aste has no 'null', for zero-shot only
                            output.append(f'[A] {at} [O] {ot} [S] {sp}')

                    elif tasks[i] == "tasd":
                        output.append(f"[A] {at} [S] {sp} [C] {ac}")

                    elif tasks[i] in ["asqp", "acos"]:
                        output.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

                    else:
                        raise NotImplementedError

                target_quads = extract_spans_para(seq=targets[i],
                                                seq_type='gold')

                if sorted(target_quads) != sorted(output_quads):
                    print("task, data:", tasks[i], datas[i])
                    print("target:", sorted(target_quads))
                    print('output:', sorted(output))
                    print("sent:", sents[i])
                    print("counter:", counter)
                    print("output quads:", output)
                    print("multi_path:", multi_outputs)
                    print()

                # if no output, use the first path
                output_str = " [SSEP] ".join(
                    output) if output else multi_outputs[0]

                outputs.append(output_str)

    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    print("pred labels count", labels_counts)

    scores, all_labels, all_preds = compute_scores(outputs,
                                                   targets,
                                                   args.paraphrase,
                                                   verbose=True)
    return scores
