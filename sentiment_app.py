import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

from models import Bidirectional_lstm
from attention import Bahdanau_Attention

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hardcoded model config ===
embed_dim = 768              # BERT base hidden size
hid_DIM = 128             # Your BiLSTM hidden dim
aTTN_HIDDEN_DIM = 64         # Attention internal size
oUTPUT_DIM = 2               # 2-class classifier
mAX_LEN = 128                # BERT max token length

# === Load BERT ===
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
bert_model.eval()

# === Load checkpoint ===
checkpoint = torch.load("checkpoint.pth", map_location=device)

# === Recreate and load model & attention ===
model = Bidirectional_lstm(embed_dim=embed_dim, hidden_dim=hid_DIM, output_dim=oUTPUT_DIM).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

attention = Bahdanau_Attention(hidden_size=2 * hid_DIM, attention_hidden_size=aTTN_HIDDEN_DIM).to(device)
attention.load_state_dict(checkpoint["attention_state_dict"])
attention.eval()

# === Streamlit UI ===
st.title("🎬 Movie Review Sentiment Classifier (BiLSTM + Bahdanau + BERT)")
st.markdown("**Architecture**: BERT embeddings → BiLSTM → Bahdanau Attention → FC")

user_input = st.text_area("📝 Enter movie review text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        with torch.no_grad():
            # Tokenize and encode
            inputs = tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=mAX_LEN)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Get BERT embeddings
            bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
            embeddings = bert_outputs.last_hidden_state  # shape: (1, MAX_LEN, 768)

            # BiLSTM forward
            lstm_out, final_hidden = model(embeddings)  # lstm_out: (1, MAX_LEN, 2*hidden)

            # Attention forward
            attn_scores, context = attention(lstm_out, final_hidden)  # context: (1, 2*hidden)

            # Classification
            logits = model.fl(context)  # (1, output_dim)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            confidence = probs[pred_label]

            st.subheader("🧠 Prediction")
            st.write(f"**Sentiment:** {'Positive 👍' if pred_label == 1 else 'Negative 👎'}")
            st.write(f"**Confidence:** `{confidence:.4f}`")

            # === Attention explanation ===
            scores = attn_scores.squeeze(0).squeeze(-1).cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            token_weights = list(zip(tokens, scores[:len(tokens)]))
            top_tokens = sorted(token_weights, key=lambda x: x[1], reverse=True)[:10]

            st.subheader("🔍 Top Influential Tokens")
            for word, score in top_tokens:
                st.markdown(f"- **{word}** — Attention Score: `{score:.4f}`")
