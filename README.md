# DA6401 Assignment-3:

## Transliteration using RNN's
### DA24M026
### Vamsi krishna Mohan
A sequence-to-sequence model implementation for transliteration between English and Indian languages (Hindi, Tamil, Telugu) with and without attention mechanisms.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Results](#results)

## Features

- **Two Model Variants**
  - With Bahdanau Attention
  - Without Attention
- **Supports Multiple Languages**
- **Beam Search Decoding**
- **Visualization Tools**
  - Attention Heatmaps
  - Word Clouds
  - Character Connectivity
- **Evaluation Metrics**
  - BLEU Score
  - Word Accuracy
  - Character Error Rate

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/vamsikrishnamohan/DA6401--Assignment-3.git


# Install dependencies
pip install -r requirements.txt
```
## Training 

### With attention
```bash
python with_attention/main.py \
  --data_dir ./data/ \
  --embedding_dim 256 \
  --units 512 \
  --batch_size 64 \
  --epochs 20 \
  --learning_rate 0.001 \
  --beam_width 3 \
  --use_wandb
  ```
 ### without attention
 ``` bash
 python without_attention/main.py \  #replace with main path 
  --data_dir ./data/ \   
  --embedding_dim 128 \
  --units 256 \
  --batch_size 128 \
  --epochs 15 \
  --learning_rate 0.001
  ```
  ## Evaluation
  ``` bash
  python evaluate.py \  #(ensure you are in either with attention or without attention code folder)
  --checkpoint path/to/model \
  --test_data ./data/hi/test.tsv \
  --output_file results.json \
  [--beam_width 5] \
  [--visualize_attention]
 ```
## Visualization
### word clouds
```bash
python -m utils.visualization --model path/to/model --num_words 50
```
### Attention Heatmaps
``` bash
from utils.visualization import plot_attention_heatmap

plot_attention_heatmap(
    attention_weights=attention_matrix,
    input_words=list("example"),
    output_words=list("उदाहरण")
)
```
## Results 
Predictions with and without attention layer are downloaded and saved in csv format available in this repo.
And all the visualization results are also in repo in - [HERE](#visualization_results) folder

