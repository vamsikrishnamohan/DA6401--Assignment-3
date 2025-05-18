# DA24M026-Assignment3

1. This notebook is structured in such a way that all the cells can be run one after another. Run All Cells command can also be used, but be wary of WandB sweeps at the end.
2. To run the model without WandB, use the following code:
```python
model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=3,
                        decoder_layers=3,
                        layer_type="lstm",
                        units=256,
                        dropout=0.2,
                        attention=False)
```
3. To run the model with WandB sweep, use the following code:
```python
# Creating the WandB config
sweep_config = {
  "name": "Sweep 1- Assignment3",
  "method": "grid",
  "parameters": {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "layer_type": {
            "values": ["rnn", "gru", "lstm"]
        }
    }
}
# Creating a sweep
sweep_id = wandb.sweep(sweep_config, project="cs6910-assignment3")
# Running the sweep
wandb.agent(sweep_id, function=lambda: train_with_wandb("hi"))
```
4. To visualize the model outputs in the form of a wordcloud, use the following code:
```python
# "model" here is the output of the function in step 2
visualize_model_outputs(model, n=20)
```
5. To visualise the model connectivity, use the following code:
```python
# Sample some words from the test data
test_words = get_test_words(5)
# Visualise connectivity for "test_words"
for word in test_words:
    visualise_connectivity(model, word, activation="scaler")
```
6. WandB report can be found at: https://wandb.ai/da24m026-indian-institute-of-technology-madras/Assignment3/reports/DA24M026-Assignment-3---VmlldzoxMjc5OTg2Ng
