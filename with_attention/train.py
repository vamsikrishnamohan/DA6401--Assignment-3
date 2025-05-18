import tensorflow as tf
from tqdm import tqdm
from utils.metrics import calculate_bleu

def train_step(model, optimizer, inp, targ, loss_fn):
    loss = 0
    hidden = model.encoder.initialize_hidden_state()
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = model.encoder(inp, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['\t']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = model.decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_fn(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = model.encoder.trainable_variables + model.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def train_model(model, dataset, val_dataset, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(args.epochs):
        total_loss = 0
        for (batch, (inp, targ)) in tqdm(enumerate(dataset)):
            batch_loss = train_step(model, optimizer, inp, targ, loss_fn)
            total_loss += batch_loss

        val_loss = evaluate_model(model, val_dataset, loss_fn)
        print(f'Epoch {epoch+1} Loss {total_loss/args.steps_per_epoch:.4f} Val Loss {val_loss:.4f}')