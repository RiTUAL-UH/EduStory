# _*_ coding: utf-8 _*_

import argparse

from tqdm import tqdm
import itertools
import time
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

import random

import pandas as pd
import numpy as np
import statistics

from sklearn.model_selection import train_test_split

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--dev_run", action="store_true")
parser.add_argument("--base_dir", type=str, default="./")
parser.add_argument("--model_save_dir", type=str, default="./save/")
parser.add_argument("--log_save_dir", type=str, default="./record/")

parser.add_argument("--use_gpu_idx", type=int, default=0)

parser.add_argument(
    "--train_batch_size", type=int, default=40, help="train_batch_size."
)
parser.add_argument("--dev_batch_size", type=int, default=2, help="dev_batch_size.")
parser.add_argument("--test_batch_size", type=int, default=1, help="test_batch_size.")


parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
parser.add_argument("--training_epochs", type=int, default=3, help="Training epochs.")
parser.add_argument("--patience", type=int, default=10, help="Early stop patience.")
parser.add_argument(
    "--multiple_runs", type=int, default=5, help="Multiple runs of experiment."
)
parser.add_argument("--numpy_seed", type=int, default=42, help="NumPy seed.")
parser.add_argument("--random_seed", type=int, default=42, help="Torch seed.")

# args = parser.parse_args()
# args = parser.parse_args(args=['--dev_run', '--training_epochs', '5'])
args = parser.parse_args(args=["--training_epochs", "3"])

device = "cuda:0"

rand_seed = args.random_seed
torch.manual_seed(rand_seed)
random.seed(0)
np.random.seed(0)

edustory = pd.read_csv("EduStory.tsv", sep="\t")

hold_out_story = edustory.loc[edustory["Generation hold out"].notna()]

edustory = edustory.loc[edustory["Generation hold out"].isnull()]

edustory = edustory.sample(frac=1, random_state=args.random_seed)

edustory = edustory.reset_index(drop=True)

edustory["idx"] = list(range(len(edustory)))

train_story = [
    "Story: " + each + "The main idea of this story is: "
    for each in list(edustory["Story"])
]

train_moral = list(edustory["Moral"])

train_data = list(zip(train_story, train_moral))

dev_story = [
    "Story: " + each + "The main idea of this story is: "
    for each in list(hold_out_story["Story"])
]

dev_moral = list(hold_out_story["Moral"])

dev_data = list(zip(dev_story, dev_moral))


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_pairs, tokenizer):
        self.tokenizer = tokenizer
        self.data = list_of_pairs
        self.source_len = 500
        self.summ_len = 50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ctext = self.data[index][0]
        text = self.data[index][1]

        source = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=self.source_len,
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.summ_len,
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

train_dataset = GenerationDataset(train_data, tokenizer)

dev_dataset = GenerationDataset(dev_data, tokenizer)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=3)

dev_loader = torch.utils.data.DataLoader(dev_dataset, shuffle=False, batch_size=1)

train_sample = iter(train_loader).next()

dev_sample = iter(dev_loader).next()


class ExperimentLog:
    def __init__(self, logfile):
        self.logfilename = logfile

    def __call__(self, content):
        with open(self.logfilename, "a+") as f_log:
            if not isinstance(content, str):
                content = "{:.4f}".format(content)
            f_log.write(content)
            f_log.write("\n")


logger = ExperimentLog(
    args.log_save_dir + "t5large_gen_seed_" + str(rand_seed) + ".txt"
)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.positive_criteria = True
        self.val_criteria_min = -1
        self.delta = delta

        self.ifsave = False
        self.new_best = False

    def __call__(self, val_criteria, model, path):
        print("val_criteria={}".format(val_criteria))
        score = val_criteria
        # initialize
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.ifsave, val_criteria, model, path)
            self.new_best = True
        # if new run worse than previous best
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.new_best = False
            if self.verbose:
                print(
                    f"No improvement. EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        # see new best, save model, reset counter
        else:
            self.best_score = score
            self.save_checkpoint(self.ifsave, val_criteria, model, path)
            self.counter = 0
            self.new_best = True

    def save_checkpoint(self, ifsave, val_criteria, model, path):
        if ifsave:
            if self.verbose:
                print(
                    f"Validation improved ({self.val_criteria_min:.6f} --> {val_criteria:.6f}).  Saving model ..."
                )
            torch.save(model.state_dict(), path + "/" + "model_checkpoint.pth")
            self.val_criteria_min = val_criteria
        else:
            pass


optimizer = AdamW(model.parameters(), lr=5e-5)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


def train_epoch(model, train_loader, device):
    model.train()
    train_epoch_loss = 0

    with tqdm(train_loader, unit=" batch") as tepoch:
        for idx, batch in enumerate(tepoch):
            y = batch["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch["source_ids"].to(device, dtype=torch.long)
            mask = batch["source_mask"].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=y_ids)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
            train_epoch_loss += loss.item()

    train_epoch_loss_avg = train_epoch_loss / (idx + 1)
    print("Epoch loss={:.4f}".format(train_epoch_loss_avg))

    return train_epoch_loss_avg


@torch.no_grad()
def eval_epoch(model, eval_loader, device):
    model.eval()
    # need tracking
    eval_epoch_loss = 0
    y_pred = []
    y_true = []

    with tqdm(eval_loader, unit=" batch") as vepoch:
        for idx, batch in enumerate(vepoch):
            y = batch["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch["source_ids"].to(device, dtype=torch.long)
            mask = batch["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(input_ids=ids, attention_mask=mask)

            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]

            # record output from each step
            y_pred += preds
            y_true += target

    return y_pred


train_all_epochs_loss = []

# training loop
for epoch in range(args.training_epochs):
    train_epoch_loss = []

    print("====== {}-th epoch ======".format(epoch))
    # logger("====== {}-th epoch ======".format(epoch))

    # if not dev run, start training
    if args.dev_run != True:
        epoch_loss = train_epoch(model, train_loader, device)


# eval

eval_epoch(model, dev_loader, device)

test_story_string = "test story."

test_story_input = "Story: " + test_story_string + "The main idea of this story is: "

test_inputs = tokenizer(test_story_input, return_tensors="pt")

test_outputs = model.generate(**test_inputs)
print(tokenizer.batch_decode(test_outputs, skip_special_tokens=True))
