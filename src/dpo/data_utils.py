import random
import json
import numpy as np
from itertools import permutations
import torch
from torch.utils.data import Dataset

def read_line_examples_from_file(data_path,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    print('Data path : ', data_path)
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            tasks.append(task_name)
            datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(tuples.split())
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels


def get_para_targets_dev(sents, labels, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    new_sents = sents 
    targets = labels
    return new_sents, targets


def get_transformed_io(data_path, data_name, data_type,  args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, args.task, args.dataset, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # low resource
    if data_type == 'train' and args.data_ratio != 1.0:
        num_sample = int(len(inputs) * args.data_ratio)
        sample_indices = random.sample(list(range(0, len(inputs))), num_sample)
        sample_inputs = [inputs[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        inputs, labels = sample_inputs, sample_labels
        print(
            f"Low resource: {args.data_ratio}, total train examples = {num_sample}")
        if num_sample <= 20:
            print("Labels:", sample_labels)

    new_inputs, targets = get_para_targets_dev(inputs, labels,
                                                   args.task, args)

    print(len(inputs), len(new_inputs), len(targets))
    return new_inputs, targets



class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 task_name,
                 data_name,
                 data_type,
                 args,
                 max_len=128):
        self.data_path = f'{args.data_path}/{task_name}/{data_name}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.data_name = data_name
        self.data_type = data_type
        self.args = args

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path,
                                                 self.data_name,
                                                 self.data_type,
                                                 self.args)
        

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = ' '.join(targets[i])

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference
            # target_max_length = 1024 if self.data_type == "test" else self.max_len

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
