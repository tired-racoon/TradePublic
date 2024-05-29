import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)

class TransformerModel(keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, output_dim, rate=0.1):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.rate = rate

        self.embedding = layers.Dense(embed_dim)
        self.pos_encoding = layers.Embedding(input_dim=10000, output_dim=embed_dim)
        self.enc_layers = [TransformerEncoder(embed_dim, num_heads, ff_dim, rate) 
                           for _ in range(num_layers)]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(output_dim)

    def call(self, inputs, training):
        seq_len = inputs.shape[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        embedded = self.embedding(inputs) + self.pos_encoding(positions)

        for i in range(self.num_layers):
            embedded = self.enc_layers[i](embedded, training)

        flattened = self.flatten(embedded)
        output = self.fc(flattened)
        return output

num_layers = 4
embed_dim = fit_size
num_heads = 8
ff_dim = 512
output_dim = 1

inputs = keras.Input(shape=(fit_size, 1))
transformer_model = TransformerModel(num_layers, embed_dim, num_heads, ff_dim, output_dim)
outputs = transformer_model(inputs)
model = keras.Model(inputs, outputs)

# Компиляция модели
model.compile(optimizer='adam', loss='mae')


