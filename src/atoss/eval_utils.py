# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import os
import sys
import logging
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
from atoss.data_utils import *
from atoss.eval_utils import *
from atoss.process import *

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    n_gold = len(gold_pt)
    n_pred = len(pred_pt)

    for i in range(len(pred_pt)):
        if pred_pt[i] == gold_pt[i]:
            n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, verbose=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = gold_seqs[i]
        pred_list = pred_seqs[i]
        if verbose and i < 10:

            print("gold ", gold_seqs[i])
            print("pred ", pred_seqs[i])
            print()

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds


def evaluate(args, model, task, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    args.data_path = f'{args.path}/data/test/{args.task}/{args.dataset}'
    sents, _ = read_line_examples_from_file(
        f'{args.data_path}/{data_type}.txt', args.lowercase)
    args.data_path
    outputs, targets, probs = [], [], []

    cache_file = os.path.join(
        args.output_dir, "result_{}_{}_{}_{}{}beam{}.pickle".format(
            args.method,
            args.model,
            data_type,
            "best_" if args.load_ckpt_name else "",
            "cd_" if args.constrained_decode else "",
            args.beam_size))
    if args.load_path_cache:
        with open(cache_file, 'rb') as handle:
            (outputs, targets, probs) = pickle.load(handle)
    else:

        dataset = ABSADataset(model.tokenizer,
                              task_name=task,
                              data_type=data_type,
                              args=args,
                              max_len=args.max_seq_length)
        data_loader = DataLoader(dataset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=2)
        device = torch.device('cuda:0')
        model.model.to(device)
        model.model.eval()
        early_stopping = False
        if args.beam_size > 1 : early_stopping=True

        for batch in tqdm(data_loader):
            # beam search
            outs = model.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                early_stopping=early_stopping,
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
    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    print("pred labels count", labels_counts)
    exp_targets = [label for label in targets for _ in range(args.beam_size)]
    scores, all_labels, all_preds = compute_scores(outputs,
                                                   exp_targets,
                                                   verbose=True)
    return scores