import tensorflow as tf
import numpy as np
from data_preparation import load_dataset, preprocess_data, create_tokenizer, prepare_sequences
from load_trained_model import load_trained_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_title(input_text, tokenizer, max_sequence_length, model):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    predicted_sequence = model.predict([padded_sequence, padded_sequence])
    predicted_token_indices = np.argmax(predicted_sequence, axis=-1)
    predicted_title = tokenizer.sequences_to_texts(predicted_token_indices)
    return predicted_title[0]

# Example usage
if __name__ == "__main__":
    input_text = "Your input text here"  # Replace with your input text
    model_path = "path/to/your/model"  # Replace with the path to your trained model
    tokenizer_path = "path/to/your/tokenizer"  # Replace with the path to your saved tokenizer
    
    preprocessed_data = preprocess_data(load_dataset('dataset.json'))
    tokenizer = create_tokenizer(preprocessed_data['texts'])
    
    max_sequence_length = preprocessed_data['max_sequence_length']
    input_vocab_size = len(tokenizer.word_index) + 1  # +1 for OOV token
    
    loaded_model = load_trained_model(model_path, input_vocab_size, max_sequence_length)
    
    generated_title = generate_title(input_text, tokenizer, max_sequence_length, loaded_model)
    print("Generated Title:", generated_title)
