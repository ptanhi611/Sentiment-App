# 🧠 Sentiment Classification using RNNs/LSTMs + Attention (IMDB Dataset)

This project implements a modular deep learning pipeline for **sentiment classification** using different RNN architectures and **four attention mechanisms**. All experiments are run on the **IMDB movie review dataset**, comparing Vanilla RNN/LSTM and Bidirectional variants.

---

## 📁 Folder Structure

```
├── attention.py            # All attention mechanism classes
├── models.py               # RNN/LSTM/BiRNN/BiLSTM model definitions
├── dataset.py              # Tokenization, vocab building, and dataloading
├── train.py                # CLI-driven training and evaluation script
├── utils.py                # Metrics, evaluation, attention visualization
├── requirements.txt        # List of dependencies
│
├── raw files/              # Original notebooks and raw development code
├── model params/           # saved model weights
├── report/                 # 📊 Evaluation JSONs + 📄 Final Report
│   ├── report.pdf
│   ├── metrics_*.json
```

---

## 🧠 Model Variants

The following 16 combinations were evaluated:

| Model Type    | Attention Mechanism                        |
|---------------|---------------------------------------------|
| Vanilla RNN   | Bahdanau, Luong Dot, Luong General, Concat |
| Vanilla LSTM  | Bahdanau, Luong Dot, Luong General, Concat |
| BiRNN         | Bahdanau, Luong Dot, Luong General, Concat |
| BiLSTM        | Bahdanau, Luong Dot, Luong General, Concat |

You can also run **any model without attention** using `--attention None`.

---

## 🚀 How to Run via CLI

Install requirements:

```bash
pip install -r requirements.txt
```

Run training & evaluation:

```bash
python train.py --model BiLSTM --attention Bahdanau
```

To run without attention:

```bash
python train.py --model VanillaRNN --attention None
```

### CLI Arguments

| Argument        | Description                             | Default |
|-----------------|------------------------------------------|---------|
| `--model`       | `VanillaRNN`, `VanillaLSTM`, `BiRNN`, `BiLSTM` | Required |
| `--attention`   | `Bahdanau`, `LuongDot`, `LuongGeneral`, `LuongConcat`, `None` | `None` |
| `--epochs`      | Number of training epochs                | 10      |
| `--batch_size`  | Batch size                               | 32      |
| `--lr`          | Learning rate                            | 1e-3    |

---

## 📊 Evaluation Metrics

Each combination is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

📁 All results are saved inside the `report/` folder as:

```
metrics_<Model>_<Attention>.json
```

---

## 🧠 Dataset Details

- **Source**: HuggingFace IMDB Dataset
- **Split**: 25,000 training / 25,000 testing
- **Task**: Sequence classification (positive/negative sentiment)
- **Tokenizer**: Custom lowercasing + alphanumeric
- **OOV Handling**: Unknown tokens default to 0
- **Padding**: Max-length based per sample

---

## 📁 Refer to `report/`

The folder contains:

- 📄 `report.pdf` → Final project report with:
  - Architecture diagrams
  - Evaluation comparison
  - Attention heatmaps
  - Key learnings & challenges

- 📊 `metrics_*.json` → Evaluation results for each model + attention combo.

---

## ⚙️ Requirements

```txt
Refer requirements.txt
```

Install via:

```bash
pip install -r requirements.txt
```

---

## ✅ Features

- Modular model + attention architecture
- CLI support for experimentation
- GPU-compatible (tested on T4 via Kaggle)
- OOV handling in test set
- Attention visualization (bonus)

---

## 👨‍💻 Author

**Rishit Mittal**  
_IIT Hyderabad | July 2025_

---

## 📌 Coming Soon (Planned)

- [ ] Config file support (YAML/JSON)
- [ ] CLI for evaluation only
- [ ] Hyperparameter sweep
