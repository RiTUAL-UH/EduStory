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

from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForMultipleChoice
from collections import Counter
from transformers import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--dev_run", action="store_true")
parser.add_argument(
    "--base_dir", type=str, default="./"
)
parser.add_argument("--model_save_dir", type=str, default="./save/")
parser.add_argument("--log_save_dir", type=str, default="./record/")

parser.add_argument("--use_gpu_idx", type=int, default=1)

parser.add_argument(
    "--train_batch_size", type=int, default=40, help="train_batch_size."
)
parser.add_argument("--dev_batch_size", type=int, default=2, help="dev_batch_size.")
parser.add_argument("--test_batch_size", type=int, default=1, help="test_batch_size.")


parser.add_argument("--lr", type=float, default=5e-5, help="learning rate.")
parser.add_argument("--training_epochs", type=int, default=30, help="Training epochs.")
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
edustory = edustory.sample(frac=1, random_state=2022)
all_label = list(edustory["Final Virtue"])


z = all_label
Counter(z)

todigit_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

edustory.iloc[0]["Final Virtue"]


# random select distractors
def get_distractors_random(num, idx, story_df):
    all_candidates = list(range(len(story_df)))
    all_candidates.remove(idx)
    # print(all_candidates)
    candidates = random.sample(all_candidates, num)

    return [story_df.iloc[each]["Theme_sum"] for each in candidates]


# select distractors from diff virtue


def get_distractors_diff_virtue(num, idx, story_df):
    virtue = story_df.iloc[idx]["Final Virtue"]
    # print(virtue)
    all_candidates = list(
        story_df[story_df["Final Virtue"] != virtue].index
    )  # diff virtue
    candidates = random.sample(all_candidates, num)
    # print(candidates)
    edustory
    return [story_df.iloc[each]["Theme_sum"] for each in candidates]


# select distractors from same virtue


def get_distractors_same_virtue(num, idx, story_df):
    virtue = story_df.iloc[idx]["Final Virtue"]

    condition1 = story_df["Final Virtue"] == virtue

    # Exclude the row with column B value equals b
    moral_to_exclude = story_df.iloc[idx]["Theme"]
    condition2 = story_df["Theme"] != moral_to_exclude

    # Combine the conditions and apply them to the DataFrame

    all_candidates = list(story_df[condition1 & condition2].index)  # same virtue
    # print(all_candidates)
    candidates = random.sample(all_candidates, num)
    # print(candidates)

    return [story_df.iloc[each]["Theme_sum"] for each in candidates]


# just index
train_story = list(range(0, 306))
dev_story = list(range(306, 360))
test_story = list(range(360, 451))


def construct_story_moral_pairs(partition, story_df, repeat, get_distractors_func):
    construct_pairs = []
    for j in range(repeat):
        for i in partition:
            # for i in range(1):
            story = story_df.iloc[i]["Story_sum"]
            moral = story_df.iloc[i]["Theme_sum"]
            # distractors
            candidates = get_distractors_func(3, i, story_df)
            # set where to place correct moral
            correct_idx = random.randrange(4)
            # insert correct moral to get all candidates
            candidates.insert(correct_idx, moral)
            construct_pairs.append(
                {
                    "sid": story_df.iloc[i]["idx"],
                    "story": story,
                    "moral": moral,
                    "candidates": candidates,
                    "correct_ans": correct_idx,
                }
            )
    return construct_pairs


chosen_distractors_func = get_distractors_same_virtue

train_pairs = construct_story_moral_pairs(
    train_story, edustory, 5, chosen_distractors_func
)
dev_pairs = construct_story_moral_pairs(dev_story, edustory, 1, chosen_distractors_func)
test_pairs = construct_story_moral_pairs(
    test_story, edustory, 1, chosen_distractors_func
)


class MultipleChoiceDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_pairs):
        self.list_of_pairs = list_of_pairs

    def __getitem__(self, item):
        one_pair = self.list_of_pairs[item]
        story_list = [one_pair["story"]] * 4
        candidates_list = one_pair["candidates"]
        correct_idx = one_pair["correct_ans"]

        return {
            "story_list": story_list,
            "candidates_list": candidates_list,
            "correct_idx": correct_idx,
        }

    def __len__(self):
        return len(self.list_of_pairs)


train_dataset = MultipleChoiceDataset(train_pairs)
dev_dataset = MultipleChoiceDataset(dev_pairs)
test_dataset = MultipleChoiceDataset(test_pairs)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(data_instance):
    stories = sum([each["story_list"] for each in data_instance], [])
    candidates = sum([each["candidates_list"] for each in data_instance], [])
    encoding = tokenizer(
        stories, candidates, return_tensors="pt", padding=True, truncation=True
    )
    multiple_choice_batch = {
        k: torch.stack([v[i : i + 4] for i in range(0, len(v), 4)]).to(device)
        for k, v in encoding.items()
    }

    correct_id_batch = torch.LongTensor(
        [each["correct_idx"] for each in data_instance]
    ).to(device)

    return {
        "multiple_choice_batch": multiple_choice_batch,
        "correct_id_batch": correct_id_batch,
    }


train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn
)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn
)


class ExperimentLog:
    def __init__(self, logfile):
        self.logfilename = logfile

    def __call__(self, content):
        with open(self.logfilename, "a+") as f_log:
            if not isinstance(content, str):
                content = "{:.4f}".format(content)
            f_log.write(content)
            f_log.write("\n")


logger = ExperimentLog(args.log_save_dir + "bert_multi_seed_" + str(rand_seed) + ".txt")


class MultipleChoiceModel(nn.Module):
    def __init__(self):
        super(MultipleChoiceModel, self).__init__()
        self.model = BertForMultipleChoice.from_pretrained("bert-base-uncased")

    def forward(self, train_sample):
        outputs = self.model(
            **train_sample["multiple_choice_batch"],
            labels=train_sample["correct_id_batch"],
        )  # batch size is 1

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


model = MultipleChoiceModel().to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


def train_epoch(model, train_loader, device):
    model.train()
    train_epoch_loss = 0

    with tqdm(train_loader, unit=" batch") as tepoch:
        for idx, batch in enumerate(tepoch):
            outputs = model(batch)
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
    eval_epoch_loss = 0
    y_pred = []
    y_true = []

    with tqdm(eval_loader, unit=" batch") as vepoch:
        for idx, batch in enumerate(vepoch):
            outputs = model(batch)
            loss = outputs[0]
            prediction_digits = torch.argmax(outputs[1], dim=1)

            # record output from each step
            eval_epoch_loss += loss.item()
            y_pred.append(prediction_digits)
            y_true.append(batch["correct_id_batch"])

    eval_epoch_loss_avg = eval_epoch_loss / (idx + 1)
    y_pred = torch.cat(y_pred, dim=0).tolist()
    y_true = torch.cat(y_true, dim=0).tolist()

    eval_acc = accuracy_score(y_true, y_pred)
    logger("eval acc:")
    logger(eval_acc)

    print("eval acc:")
    print(eval_acc)

    return eval_epoch_loss_avg, eval_acc


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
    val_epoch_loss_avg, val_acc = eval_epoch(model, dev_loader, device)

    # ==================early stopping======================
    if args.dev_run != True:
        early_stopping(val_criteria=val_acc, model=model, path=args.model_save_dir)

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
test_epoch_loss_avg, test_acc = eval_epoch(model, test_loader, device)
