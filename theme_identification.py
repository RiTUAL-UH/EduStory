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
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--dev_run", action="store_true")
parser.add_argument("--model_save_dir", type=str, default="./save_model/")
parser.add_argument("--log_save_dir", type=str, default="./record/")

parser.add_argument("--use_gpu_idx", type=int, default=0)

parser.add_argument(
    "--train_batch_size", type=int, default=40, help="train_batch_size."
)
parser.add_argument("--dev_batch_size", type=int, default=2, help="dev_batch_size.")
parser.add_argument("--test_batch_size", type=int, default=1, help="test_batch_size.")
parser.add_argument("--cls_output_size", type=int, default=14, help="class num.")


parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
parser.add_argument("--training_epochs", type=int, default=20, help="Training epochs.")
parser.add_argument("--patience", type=int, default=10, help="Early stop patience.")
parser.add_argument(
    "--multiple_runs", type=int, default=5, help="Multiple runs of experiment."
)
parser.add_argument("--numpy_seed", type=int, default=42, help="NumPy seed.")
parser.add_argument("--random_seed", type=int, default=42, help="Torch seed.")


# args = parser.parse_args()

# args = parser.parse_args(args=['--dev_run', '--training_epochs', '5'])
args = parser.parse_args(args=["--training_epochs", "200"])

device = "cuda:0"

rand_seed = args.random_seed
torch.manual_seed(rand_seed)
random.seed(0)
np.random.seed(0)

edustory = pd.read_csv("EduStory.tsv", sep="\t")

virtue_or_strength = "Final Strength"

count_freq = dict(edustory[virtue_or_strength].value_counts())

count_freq

edustory["count_freq"] = edustory[virtue_or_strength]
edustory["count_freq"] = edustory["count_freq"].map(count_freq)

edustory.head(3)

edustory_morethanthree = edustory[edustory.count_freq > 3]

all_label = list(edustory_morethanthree[virtue_or_strength])


plt.hist(all_label)
plt.show()


Counter(all_label)

all_label_set = sorted(list(set(all_label)))
all_label_digit = list(range(len(all_label_set)))

args.cls_output_size = len(all_label_digit)

# todigit_dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

todigit_dic = dict(zip(all_label_set, all_label_digit))

all_texts = list(edustory_morethanthree["Story"])
all_labels = [
    todigit_dic[each] for each in list(edustory_morethanthree[virtue_or_strength])
]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    all_texts, all_labels, test_size=0.2, stratify=all_labels
)
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    train_texts, train_labels, test_size=0.15, stratify=train_labels
)

len(dev_texts)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


class ExperimentLog:
    def __init__(self, logfile):
        self.logfilename = logfile

    def __call__(self, content):
        with open(self.logfilename, "a+") as f_log:
            if not isinstance(content, str):
                content = "{:.4f}".format(content)
            f_log.write(content)
            f_log.write("\n")


logger = ExperimentLog(args.log_save_dir + "bert_cls_seed_" + str(rand_seed) + ".txt")


class ClassifierModel(nn.Module):
    def __init__(self, class_num):
        super(ClassifierModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=class_num
        )

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0):
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
                print(f"Current best dev criteria: {self.best_score}")
            if self.counter >= self.patience:
                self.early_stop = True
        # see new best, save model, reset counter
        else:
            self.best_score = score
            self.save_checkpoint(self.ifsave, val_criteria, model, path)
            self.counter = 0
            self.new_best = True

    def save_checkpoint(self, ifsave, val_criteria, model, path):
        print(
            f"Validation criteria improved ({self.val_criteria_min:.6f} --> {val_criteria:.6f})."
        )
        if ifsave:
            if self.verbose:
                print("Saving model ...")
            torch.save(model.state_dict(), path + "/" + "model_checkpoint.pth")
            self.val_criteria_min = val_criteria
        else:
            pass


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


model = ClassifierModel(class_num=args.cls_output_size).to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

train_dataset = StoryDataset(train_encodings, train_labels)
dev_dataset = StoryDataset(dev_encodings, dev_labels)
test_dataset = StoryDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train_epoch(model, train_loader, device):
    model.train()
    train_epoch_loss = 0

    with tqdm(train_loader, unit=" batch") as tepoch:
        for idx, batch in enumerate(tepoch):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            prediction_digits = torch.argmax(outputs[1], dim=1)

            # record output from each step
            eval_epoch_loss += loss.item()
            y_pred.append(prediction_digits)
            y_true.append(labels)

    eval_epoch_loss_avg = eval_epoch_loss / (idx + 1)
    y_pred = torch.cat(y_pred, dim=0).tolist()
    y_true = torch.cat(y_true, dim=0).tolist()
    # print(y_pred)
    # print(y_true)

    eval_cls_report = classification_report(y_true, y_pred, digits=4)
    logger(eval_cls_report)

    eval_f1 = f1_score(y_true, y_pred, average="macro")
    logger("eval cls f1:")
    logger(eval_f1)

    print("eval cls f1:")

    return eval_epoch_loss_avg, eval_f1


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
    val_epoch_loss_avg, val_f1 = eval_epoch(model, dev_loader, device)

    # ==================early stopping======================
    if args.dev_run != True:
        early_stopping(val_criteria=val_f1, model=model, path=args.model_save_dir)

        if early_stopping.new_best:
            best_sd = copy.deepcopy(model.state_dict())

        if early_stopping.early_stop:
            print("Early stopping at {}-th epoch.".format(epoch))
            break

# =====================test============================#
print("======== test begins =======")
# logger("======== test result =======")

# load best early stop state dict
model.load_state_dict(best_sd)
test_epoch_loss_avg, test_f1 = eval_epoch(model, test_loader, device)
