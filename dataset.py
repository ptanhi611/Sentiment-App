

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from datasets import load_dataset
import json
from transformers import BertTokenizer




# -------------------------
# Custom Dataset Class
# -------------------------

class IMDBDataset(Dataset):
    def __init__(self, X, Y,attention_masks):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.A = torch.tensor(attention_masks, dtype= torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.A[index]

# -------------------------
# Full Data Processing Pipeline
# -------------------------

def load_imdb_data(batch_size=32, max_samples=25000):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_dataset("imdb")

    # Prepare Training Data
    df_train = pd.DataFrame(data["train"]).sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Auto compute max length from tokenized data
    max_len = df_train["text"].apply(lambda x: len(tokenizer.tokenize(x))).max()
    print(f"Auto-detected max_len: {max_len}")

    # Tokenize the train data
    encoded_train = tokenizer(list(df_train["text"]),
                              padding="max_length",
                              truncation=True,
                              max_length=max_len,
                              return_tensors="pt")

    X_train = encoded_train["input_ids"]
    Y_train = torch.tensor(df_train["label"].tolist())
    attention_masks_train = encoded_train["attention_mask"]

    train_loader = DataLoader(IMDBDataset(X_train, Y_train, attention_masks_train),
                              batch_size=batch_size, shuffle=True)

    # Prepare Test Data
    df_test = pd.DataFrame(data["test"]).sample(n=max_samples, random_state=42).reset_index(drop=True)

    encoded_test = tokenizer(list(df_test["text"]),
                             padding="max_length",
                             truncation=True,
                             max_length=max_len,
                             return_tensors="pt")

    X_test = encoded_test["input_ids"]
    Y_test = torch.tensor(df_test["label"].tolist())
    attention_masks_test = encoded_test["attention_mask"]

    test_loader = DataLoader(IMDBDataset(X_test, Y_test, attention_masks_test),
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, max_len


if __name__ == "__main__":
    load_imdb_data()
