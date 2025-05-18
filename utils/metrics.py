from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, hypothesis):
    smoother = SmoothingFunction().method2
    return sentence_bleu([reference], hypothesis, smoothing_function=smoother)

def transliteration_accuracy(true_labels, pred_labels):
    correct = sum([1 for t, p in zip(true_labels, pred_labels) if t == p])
    return correct / len(true_labels)

def calculate_wer(true_words, pred_words):
    # Word Error Rate implementation
    errors = sum(1 for t, p in zip(true_words, pred_words) if t != p)
    return errors / len(true_words)