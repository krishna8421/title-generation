import tensorflow as tf
# from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.lstm_layers = [tf.keras.layers.LSTM(d_model, return_sequences=True) for _ in range(num_layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)
        return x

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.lstm_layers = [tf.keras.layers.LSTM(d_model, return_sequences=True) for _ in range(num_layers)]

    def call(self, inputs, encoder_output):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)
        return x