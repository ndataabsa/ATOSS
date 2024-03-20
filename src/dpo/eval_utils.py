import re
import os
import pandas as pd
import numpy as np
from data_utils import ABSADataset, read_line_examples_from_file


def compute_scores(pred_seqs, gold_seqs, verbose=True):

    # assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
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


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    n_gold += len(gold_pt)
    n_pred += len(pred_pt)

    for i, pred in enumerate(pred_pt):
        if pred_pt[i] == gold_pt[i]:
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
    print(scores)
    return scores

