# Library used for fine tuning
# Pandas Dataframe Library
import json

import pandas as pd
# HateBert Libarary
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")


def load_data():
    # Open train jsonl file
    with open('train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f.readlines()]

    # Create a pandas DataFrame from the parsed json data
    train_df = pd.DataFrame(train_data)

    # Open validation jsonl file
    with open('val.jsonl', 'r') as f:
        val_data = [json.loads(line) for line in f.readlines()]

    # Create a pandas DataFrame from the parsed json data
    val_df = pd.DataFrame(val_data)

    # Open test jsonl file
    with open('test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f.readlines()]

    # Create a pandas DataFrame from the parsed json data
    test_df = pd.DataFrame(test_data)

    # Print the DataFrames
    print("TRAIN")
    print(train_df.head(1))
    print("VAL")
    print(val_df.head(1))
    print("TEST")
    print(test_df.head(1))

    return train_data, val_data, test_data


def tokenize_data(data):
    tokenized_data = tokenizer(
        data["context"],
        data["target"],
        padding="max_length",
        max_length=128,
        truncation=True
    )
    tokenized_data["label"] = int(data["label"])
    return tokenized_data


def load_tokenized_data():
    train_data, val_data, test_data = load_data()
    tokenized_train = [tokenize_data(data) for data in train_data]
    tokenized_val = [tokenize_data(data) for data in val_data]
    tokenized_test = [tokenize_data(data) for data in test_data]
    return tokenized_train, tokenized_val, tokenized_test


tokenized_train_data, tokenized_val_data, tokenized_test_data = load_tokenized_data()

# load pre-trained HateBert
model = AutoModelForMaskedLM.from_pretrained("GroNLP/hateBERT")


# Define a custom dataset class
class TokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(self.data[0])

        for dictionary in self.data:
            item = {key: torch.tensor(val[idx]) for key, val in dictionary.items()}
        return item


# Define the training and validation datasets

train_dataset = TokenizedDataset(tokenized_train_data)
val_dataset = TokenizedDataset(tokenized_val_data)

# Define the training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
    evaluation_strategy='epoch',  # evaluation strategy to adopt during training
    disable_tqdm=False  # disable tqdm progress bar
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()
