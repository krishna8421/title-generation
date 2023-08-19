import tensorflow as tf
from transformer import TransformerEncoder, TransformerDecoder

def build_model(input_vocab_size, target_vocab_size, mha_output_size, max_sequence_length):
    encoder_input = tf.keras.layers.Input(shape=(None,))
    decoder_input = tf.keras.layers.Input(shape=(None,))

    print("Encoder Input Shape:", encoder_input.shape)
    print("Decoder Input Shape:", decoder_input.shape)

    # Create encoder and decoder embeddings
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, mha_output_size)(encoder_input)
    decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, mha_output_size)(decoder_input)

    print("Encoder Embedding Shape:", encoder_embedding.shape)
    print("Decoder Embedding Shape:", decoder_embedding.shape)

    # Build Transformer-based encoder and decoder
    encoder_output = TransformerEncoder(num_layers=4, d_model=mha_output_size, num_heads=8)(encoder_embedding)
    decoder_output = TransformerDecoder(num_layers=4, d_model=mha_output_size, num_heads=8)(decoder_embedding, encoder_output)

    print("Encoder Output Shape:", encoder_output.shape)
    print("Decoder Output Shape:", decoder_output.shape)


    output = tf.keras.layers.Dense(target_vocab_size, activation='softmax')(decoder_output)

    print("Output Shape:", output.shape)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output)

    print(model)

    return model


def train_model(model, encoder_input, decoder_input, target_output, save_path):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input, decoder_input], target_output, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the trained model to the specified path
    model.save(save_path)

# Example usage
# if __name__ == "__main__":
#     input_vocab_size = 10000  # Replace with your desired value
#     target_vocab_size = 8000  # Replace with your desired value
#     mha_output_size = 256  # Replace with your desired value
    
#     encoder_input = ...  # Replace with your encoded input sequences
#     decoder_input = ...  # Replace with your encoded decoder input sequences
#     target_output = ...  # Replace with your one-hot encoded target output sequences
    
#     model = build_model(input_vocab_size, target_vocab_size, mha_output_size)
#     save_path = 'trained_model.h5'  # Replace with the desired path and filename
#     train_model(model, encoder_input, decoder_input, target_output, save_path)
