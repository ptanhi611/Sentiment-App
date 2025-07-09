import streamlit as st
import torch
import torch.nn.functional as F
import os
import json
import numpy as np

from models import Bidirectional_lstm
from attention import Bahdanau_Attention
from dataset import tokenizer, encode, padding  # using your dataset functions

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_CONFIG_PATH = os.path.join(BASE_DIR, "vocab_config.json")

MODEL_PATH = os.path.join(BASE_DIR, "model params", "Model_params", "model_BiLSTM_Bahdanau.pt")
EMBED_PATH = os.path.join(BASE_DIR, "model params", "Embedding Layer Params", "embed_BiLSTM_Bahdanau.pt")
ATTEN_PATH = os.path.join(BASE_DIR, "model params", "Attention param", "attention_BiLSTm_Bahdanau.pt")

# === Load vocab & max_len ===
with open(VOCAB_CONFIG_PATH, "r") as f:
    config = json.load(f)

vocab = config["vocab"]
max_len = config["max_len"]

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model config ===
embed_dim = 100
model_hidden_size = 128
attention_hidden_size = 64  
output_size = 2
bidir = True

# === Load model components ===
embedding_layer = torch.nn.Embedding(num_embeddings=len(vocab) + 1, embedding_dim=embed_dim, padding_idx=0).to(device)
embedding_layer.load_state_dict(torch.load(EMBED_PATH, map_location=device))
embedding_layer.eval()

model = Bidirectional_lstm(embed_dim, model_hidden_size, output_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

attention = Bahdanau_Attention(model_hidden_size *2 if bidir else model_hidden_size, attention_hidden_size).to(device)
attention.load_state_dict(torch.load(ATTEN_PATH, map_location=device))
attention.eval()

# === Streamlit UI ===
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Using model")

user_input = st.text_area("📝 Your Review", placeholder="Type your movie review here...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        tokens = tokenizer(user_input)
        encoded = encode(tokens, vocab)
        padded = padding(encoded, max_len)
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)

        with torch.no_grad():
            embedded = embedding_layer(input_tensor)
            output_seq, final_hidden = model(embedded)
            scores, context = attention(output_seq, final_hidden)
            logits = model.fl(context)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            confidence = probs[pred_label]

            st.subheader("🧠 Prediction")
            st.write(f"**Sentiment:** {'Positive 👍' if pred_label == 1 else 'Negative 👎'}")
            st.write(f"**Confidence:** {confidence:.4f}")

            # === Attention Explanation ===
            scores = scores.squeeze(0).squeeze(-1).cpu().numpy()
            token_weights = list(zip(tokens, scores[:len(tokens)]))
            top_tokens = sorted(token_weights, key=lambda x: x[1], reverse=True)[:10]

            st.subheader("🔍 Top Influential Words")
            for word, score in top_tokens:
                st.markdown(f"- **{word}** — Attention Score: `{score:.4f}`")
