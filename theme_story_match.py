import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    util,
    models,
    evaluation,
    losses,
    InputExample,
)
import logging
from datetime import datetime
from torch.utils.data import Dataset
import random
import argparse

from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import numpy as np
import torch
from statistics import mean

parser = argparse.ArgumentParser()

parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--max_seq_length", default=500, type=int)
parser.add_argument("--model_name", required=False)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used? Multiple systems seperated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=100, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=True, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--device", default="cuda:1", type=str)

args = parser.parse_args([])

theme_story_setting = False

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

# fine-tune
model_name = "sentence-transformers/msmarco-bert-base-dot-v5"

train_batch_size = args.train_batch_size
max_seq_length = args.max_seq_length
ce_score_margin = args.ce_score_margin
num_negs_per_system = args.num_negs_per_system
num_epochs = args.epochs

# Load model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name, device=args.device)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), args.pooling
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=args.device
    )

model_save_path = "save_model/train_bi-encoder-{}-{}".format(
    model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

device = "cuda:0"

rand_seed = 42
torch.manual_seed(rand_seed)
random.seed(0)
np.random.seed(0)

all_story = pd.read_csv("Edustory.tsv", sep="\t")
all_story = all_story.sample(frac=1, random_state=2022)

if theme_story_setting:
    all_story = all_story.rename(columns={"Story": "Theme", "Theme": "Story"})

train_story = all_story.iloc[:306, :]
dev_story = all_story.iloc[306:360, :]
test_story = all_story.iloc[360:, :]

train_queries = dict(zip(train_story["idx"], train_story["Story"]))
train_corpus = dict(zip(train_story["idx"], train_story["Theme"]))


def reject_sample(lst, exception):
    while True:
        choice = random.choice(lst)
        if choice != exception:
            return choice


# use MSMARCO dataset format
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query

        pos_id = self.queries_ids[item]  # Pop positive and add at end
        pos_text = self.corpus[pos_id]

        neg_id = reject_sample(self.queries_ids, pos_id)
        neg_text = self.corpus[neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=train_corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

from sentence_transformers import evaluation

dev_story_list = list(dev_story["Story"])
dev_moral_list = list(dev_story["Theme"])
scores = [1.0] * len(dev_story_list)
labels = [1] * len(dev_story_list)

# evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_story_list, dev_moral_list, scores)
evaluator = evaluation.BinaryClassificationEvaluator(
    dev_story_list, dev_moral_list, labels, batch_size=16, show_progress_bar=True
)
# ... Your other code to load training data

# # Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    # epochs=num_epochs,
    epochs=5,
    warmup_steps=args.warmup_steps,
    use_amp=True,
    # evaluator=evaluator,
    # checkpoint_path=model_save_path,
    # checkpoint_save_steps=len(train_dataloader),
    optimizer_params={"lr": args.lr},
)


def compute_mrr(rankings):
    reciprocal_ranks = [
        1 / (rank + 1) for rank in rankings
    ]  # Add 1 to each rank to avoid division by zero
    return mean(reciprocal_ranks)


def custom_eval(model, test_df):
    test_story_list = list(test_df["Story"])
    test_moral_list = list(test_df["Theme"])
    test_story_vec = model.encode(test_story_list, convert_to_tensor=True)
    test_moral_vec = model.encode(test_moral_list, convert_to_tensor=True)
    test_similarity_scores = util.dot_score(test_story_vec, test_moral_vec)
    test_np_similarity_scores = test_similarity_scores.cpu().numpy()
    y = (-test_np_similarity_scores).argsort()  # get rank value of each element
    all_pos = []  # what rank is at each position
    for i in range(len(y)):
        pos = list(y[i]).index(i)
        all_pos.append(pos)
    avg_rank = np.mean(all_pos)
    mrr = compute_mrr(all_pos)

    return avg_rank, mrr


dev_rank = custom_eval(model, dev_story)
test_rank = custom_eval(model, test_story)
print("dev rank: ", dev_rank, "test rank: ", test_rank)
