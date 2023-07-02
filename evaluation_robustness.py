"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
import csv
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.utils.prune as prune
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)

import sys
from utils import Collator, Huggingface_dataset, ExponentialMovingAverage
from masked_bert import MaskedBertForSequenceClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_path', type=str, default='./your_search-ticket_path')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--num_labels', type=int, default=2)

    # adversarial attack
    parser.add_argument("--num_examples", default=872, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--result_file', type=str, default='attack_result.csv')
    parser.add_argument('--out_w_per_mask', type=int, default=1)
    parser.add_argument('--in_w_per_mask', type=float, default=1)
    parser.add_argument('--mask_p', type=float, default=0.9)  # init mask score

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def adversarial_attack(args):

    attack_path = args.model_path
    original_accuracy, accuracy_under_attack, attack_succ = attack_test(attack_path, args)

    out_csv = open(args.result_file, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerow([attack_path, original_accuracy, accuracy_under_attack, attack_succ])
    out_csv.close()


def attack_test(attack_path, args):

    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
    from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
    from textattack import Attacker
    from textattack import AttackArgs

    # for model
    config = AutoConfig.from_pretrained(attack_path)
    model = MaskedBertForSequenceClassification.from_pretrained(
        attack_path, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.cuda()
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        attack_valid = 'test'
    else:
        attack_valid = 'validation'

    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)

    # for attack
    attack_args = AttackArgs(num_examples=args.num_examples,
                             disable_stdout=True, random_seed=args.seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0

    return original_accuracy, accuracy_under_attack, attack_succ


def main(args):
    set_seed(args.seed)
    adversarial_attack(args)


if __name__ == '__main__':

    args = parse_args()
    level = logging.INFO
    logging.basicConfig(level=level)

    main(args)

