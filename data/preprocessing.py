from typing import Tuple
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def add_special_tokens(df: pd.DataFrame, 
                      cols: list = ['input', 'target'],
                      sos: str = '\t',
                      eos: str = '\n') -> pd.DataFrame:
    """Add start/end tokens to sequences"""
    df = df.copy()
    for col in cols:
        df[col] = sos + df[col].astype(str) + eos
    return df

def create_tokenizers(train_df: pd.DataFrame,
                     input_col: str = 'input',
                     target_col: str = 'target') -> Tuple[Tokenizer, Tokenizer]:
    """Create character-level tokenizers"""
    input_tokenizer = Tokenizer(char_level=True, filters='')
    target_tokenizer = Tokenizer(char_level=True, filters='')
    
    input_tokenizer.fit_on_texts(train_df[input_col])
    target_tokenizer.fit_on_texts(train_df[target_col])
    
    return input_tokenizer, target_tokenizer

def tokenize_dataset(df: pd.DataFrame,
                    input_tokenizer: Tokenizer,
                    target_tokenizer: Tokenizer,
                    max_seq_length: int = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convert text sequences to padded token indices"""
    input_sequences = input_tokenizer.texts_to_sequences(df['input'])
    target_sequences = target_tokenizer.texts_to_sequences(df['target'])
    
    input_padded = pad_sequences(input_sequences, padding='post', maxlen=max_seq_length)
    target_padded = pad_sequences(target_sequences, padding='post', maxlen=max_seq_length)
    
    return tf.convert_to_tensor(input_padded, dtype=tf.int32), \
           tf.convert_to_tensor(target_padded, dtype=tf.int32)

def create_tf_dataset(inputs: tf.Tensor,
                     targets: tf.Tensor,
                     batch_size: int = 128,
                     shuffle: bool = True) -> tf.data.Dataset:
    """Create TensorFlow Dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(inputs))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)