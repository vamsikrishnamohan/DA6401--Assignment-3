# Training loop for non-attention model
import tensorflow as tf
from tqdm import tqdm

def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(model, optimizer, loss_object, inp, targ):
    loss = 0
    hidden = model.encoder.initialize_hidden_state()
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = model.encoder(inp, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([1] * model.encoder.batch_size, 1)  # Assuming 1 is <start>

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = model.decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions, loss_object)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = model.encoder.trainable_variables + model.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def train_model(model, train_dataset, val_dataset, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    for epoch in range(args.epochs):
        total_loss = 0
        for (batch, (inp, targ)) in tqdm(enumerate(train_dataset)):
            batch_loss = train_step(model, optimizer, loss_object, inp, targ)
            total_loss += batch_loss

        # Validation
        val_loss = 0
        for (batch, (inp, targ)) in enumerate(val_dataset):
            loss = 0
            hidden = model.encoder.initialize_hidden_state()
            enc_output, enc_hidden = model.encoder(inp, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([1] * args.batch_size, 1)
            
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = model.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions, loss_object)
                dec_input = tf.expand_dims(targ[:, t], 1)
            
            val_loss += loss / int(targ.shape[1])

        print(f'Epoch {epoch+1} Loss {total_loss/len(train_dataset):.4f} Val Loss {val_loss/len(val_dataset):.4f}')