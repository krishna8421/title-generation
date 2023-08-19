import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def preprocess_data(dataset):
    texts = [data['content'] for data in dataset]
    titles = [data['title'] for data in dataset]
    
    # Calculate sequence lengths
    sequence_lengths = [len(content.split()) for content in texts]
    max_sequence_length = max(sequence_lengths)
    
    return {'texts': texts, 'titles': titles, 'sequence_lengths': sequence_lengths, 'max_sequence_length': max_sequence_length}

def create_tokenizer(texts):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def prepare_sequences(texts, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    return padded_sequences

# Example usage
if __name__ == "__main__":
    dataset = load_dataset('dataset.json')
    preprocessed_data = preprocess_data(dataset)
    tokenizer = create_tokenizer(preprocessed_data['texts'])
    title_tokenizer = create_tokenizer(preprocessed_data['titles'])  # Create separate tokenizer for titles
    sequences = prepare_sequences(preprocessed_data['texts'], tokenizer, preprocessed_data['max_sequence_length'])
    title_sequences = prepare_sequences(preprocessed_data['titles'], title_tokenizer, preprocessed_data['max_sequence_length'])
    print("Texts:", preprocessed_data['texts'][:2])
    print("Titles:", preprocessed_data['titles'][:2])
    print("Sequence Lengths:", preprocessed_data['sequence_lengths'][:2])
    print("Max Sequence Length:", preprocessed_data['max_sequence_length'])
    print("Tokenizer Vocabulary Size:", len(tokenizer.word_index))
    print("Title Tokenizer Vocabulary Size:", len(title_tokenizer.word_index))
    print("Padded Sequences Shape:", sequences.shape)
    print("Title Sequences Shape:", title_sequences.shape)
