import argparse
import torch

from models import get_model_class
from attention import get_attention_class
from dataset import load_imdb_data
from utils import train_and_evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Train Sentiment Model with Attention")
    parser.add_argument('--model', type=str, choices=['VanillaRNN', 'VanillaLSTM', 'BiRNN', 'BiLSTM'], required=True)
    parser.add_argument('--attention', type=str, choices=['Bahdanau', 'LuongDot', 'LuongGeneral', 'LuongConcat', 'None'], default='None')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Loading IMDB Dataset...")
    train_loader, test_loader, vocab, max_len = load_imdb_data(batch_size=args.batch_size)

    embed_dim = 100
    hidden_size = 128
    bidirectional = args.model in ['BiRNN', 'BiLSTM']
    model_hidden_size = hidden_size * 2 if bidirectional else hidden_size
    attention_hidden_size = 64
    output_size = 2  # Binary classification

    model_class = get_model_class(args.model)
    model = model_class(embed_dim, hidden_size, output_size).to(device)

    if args.attention != 'None':
        print(f"🧠 Using Attention: {args.attention}")
        attention_class = get_attention_class(args.attention)
        attention = attention_class(model_hidden_size, attention_hidden_size).to(device)
    else:
        print("🚫 No Attention being used.")
        attention = None

    print(f"\n🚀 Training Model: {args.model} | Attention: {args.attention}\n")
    results = train_and_evaluate(model, attention, train_loader, test_loader, vocab, args)

    print("\n📊 Final Evaluation Metrics:\n")
    for key, value in results.items():
        print(f"{key:20s}: {value}")


if __name__ == "__main__":
    main()
