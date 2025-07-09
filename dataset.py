

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from datasets import load_dataset
import json

# -------------------------
# Tokenizer and Preprocessing
# -------------------------

def tokenizer(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().split()

def encode(tokens, vocab):
    return [vocab.get(token, 0) for token in tokens]  # 0 for unknown

def padding(tokens, max_len):
    return tokens + [0] * (max_len - len(tokens))


# -------------------------
# Custom Dataset Class
# -------------------------

class IMDBDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# -------------------------
# Full Data Processing Pipeline
# -------------------------

def load_imdb_data(batch_size=32, max_samples=25000):
    data = load_dataset("imdb")

    # Prepare Training Data
    df_train = pd.DataFrame(data["train"]).sample(n=max_samples, random_state=42).reset_index(drop=True)
    df_train["tokenized"] = df_train["text"].apply(tokenizer)
    
    # Build vocab
    all_tokens = [tok for row in df_train["tokenized"] for tok in row]
    
    vocab = {word: i+1 for i, word in enumerate(set(all_tokens))}  # NO <PAD>, starts from 1

# Compute max length
    max_len = max([len(row) for row in df_train["tokenized"]])

    print("creating json file")
    with open("vocab_config.json", "w") as f:
        json.dump({"vocab": vocab, "max_len": max_len}, f)


    # df_train["encoded"] = df_train["tokenized"].apply(lambda x: encode(x, vocab))
    # df_train["padded"] = df_train["encoded"].apply(lambda x: padding(x, max_len))

    # X_train = df_train["padded"].tolist()
    # Y_train = df_train["label"].tolist()

    # train_loader = DataLoader(IMDBDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

    # # Prepare Test Data
    # df_test = pd.DataFrame(data["test"]).sample(n=max_samples, random_state=42).reset_index(drop=True)
    # df_test["tokenized"] = df_test["text"].apply(tokenizer)
    # df_test["encoded"] = df_test["tokenized"].apply(lambda x: encode(x, vocab))
    # df_test["padded"] = df_test["encoded"].apply(lambda x: padding(x, max_len))

    # X_test = df_test["padded"].tolist()
    # Y_test = df_test["label"].tolist()

    # test_loader = DataLoader(IMDBDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    # return train_loader, test_loader, vocab, max_len


if __name__ == "__main__":
    load_imdb_data()
