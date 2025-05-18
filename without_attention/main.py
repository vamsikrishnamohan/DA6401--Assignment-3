import argparse
from data import load_tsv, preprocess_data, create_tokenizers
from model import Encoder, Decoder
from train import train_model
from evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train transliteration model without attention')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_target_len', type=int, default=25)
    return parser.parse_args()

class Seq2SeqModel:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

if __name__ == '__main__':
    args = parse_args()
    
    # Load and preprocess data
    train_data = load_tsv(f"{args.data_dir}/train.tsv")
    val_data = load_tsv(f"{args.data_dir}/dev.tsv")
    test_data = load_tsv(f"{args.data_dir}/test.tsv")
    
    # Create model
    encoder = Encoder(
        vocab_size=len(input_tokenizer.word_index)+1,
        embedding_dim=args.embedding_dim,
        enc_units=args.units,
        batch_size=args.batch_size
    )
    decoder = Decoder(
        vocab_size=len(target_tokenizer.word_index)+1,
        embedding_dim=args.embedding_dim,
        dec_units=args.units,
        batch_size=args.batch_size
    )
    model = Seq2SeqModel(encoder, decoder)
    
    # Train
    train_model(model, train_data, val_data, args)
    
    # Evaluate
    bleu_score, accuracy = evaluate_model(model, test_data, target_tokenizer, args)
    print(f"Test BLEU: {bleu_score:.4f}, Accuracy: {accuracy:.4f}")