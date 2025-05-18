import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def compute_gradients(model, input_seq):
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        outputs = model(input_seq)
    return tape.gradient(outputs, input_seq)

def visualize_character_connectivity(gradients, input_chars, output_chars):
    scaler = MinMaxScaler()
    normalized_grads = scaler.fit_transform(gradients.numpy())
    
    plt.figure(figsize=(12, 6))
    plt.imshow(normalized_grads, cmap='viridis', aspect='auto')
    plt.yticks(range(len(output_chars)), output_chars)
    plt.xticks(range(len(input_chars)), input_chars)
    plt.colorbar()
    plt.show()

def get_activation_maps(model, layer_name, input_data):
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    return intermediate_model.predict(input_data)