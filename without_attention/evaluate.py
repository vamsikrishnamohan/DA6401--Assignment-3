# Evaluation metrics and logic
import numpy as np
from utils.metrics import calculate_bleu, transliteration_accuracy
from utils.visualization import plot_attention_heatmap

def evaluate_model(model, test_dataset, tokenizer, args):
    total_bleu = 0
    total_acc = 0
    attention_weights_list = []
    
    for (batch, (inp, targ)) in enumerate(test_dataset):
        batch_bleu = 0
        batch_acc = 0
        hidden = model.encoder.initialize_hidden_state()
        enc_output, enc_hidden = model.encoder(inp, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([1] * args.batch_size, 1)
        
        predictions = []
        attention_weights = []
        
        for t in range(args.max_target_len):
            preds, dec_hidden, attn = model.decoder(
                dec_input, dec_hidden, enc_output)
            predicted_ids = tf.argmax(preds, axis=1)
            predictions.append(predicted_ids.numpy())
            attention_weights.append(attn.numpy())
            dec_input = tf.expand_dims(predicted_ids, 1)
        
        # Post-process predictions
        predictions = np.array(predictions).T
        for i in range(predictions.shape[0]):
            pred_seq = [tokenizer.index_word[id] for id in predictions[i] if id != 0]
            true_seq = [tokenizer.index_word[id] for id in targ[i].numpy() if id != 0]
            
            total_bleu += calculate_bleu(true_seq, pred_seq)
            total_acc += transliteration_accuracy(true_seq, pred_seq)
        
        attention_weights_list.append(attention_weights)
    
    avg_bleu = total_bleu / len(test_dataset)
    avg_acc = total_acc / len(test_dataset)
    
    # Visualize attention for first sample
    plot_attention_heatmap(
        attention_weights_list[0][0], 
        inp_text=[tokenizer.index_word[id] for id in inp[0].numpy()], 
        pred_text=[tokenizer.index_word[id] for id in predictions[0]]
    )
    
    return avg_bleu, avg_acc