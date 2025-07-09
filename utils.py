# utils.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json

def train_and_evaluate(model, attention, train_loader, test_loader, vocab, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if attention:
        attention = attention.to(device)

    embedding_layer = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=100, padding_idx=0).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding_layer.parameters()) +
                                 (list(attention.parameters()) if attention else []), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            X_embedded = embedding_layer(X_batch)
            model_output, hidden = model(X_embedded)

            if attention:
                scores, context = attention(model_output, hidden)
                output = model.fl(context)
            else:
                output = model.fl(hidden)

            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

    return evaluate_model(model, attention, test_loader, embedding_layer, args)

def evaluate_model(model, attention, test_loader, embedding_layer, args):
    model.eval()
    if attention: attention.eval()

    y_true, y_pred = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_embedded = embedding_layer(X_batch)

            model_output, hidden = model(X_embedded)
            if attention:
                scores, context = attention(model_output, hidden)
                final_output = model.fl(context)
            else:
                final_output = model.fl(hidden)

            predictions = torch.argmax(final_output, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(recall, 5),
        "f1_score": round(f1, 5),
        "confusion_matrix": cm.tolist()
    }

    print("\n📊 Evaluation Metrics:")
    print(json.dumps(results, indent=4))

    return results
