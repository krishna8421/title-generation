import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from data_preparation import load_dataset, preprocess_data, create_tokenizer, prepare_sequences
from model_training import build_model, train_model
from inference import generate_title
from load_trained_model import load_trained_model

# Load and preprocess the dataset
print("Loading and preprocessing the dataset...")
dataset = load_dataset('dataset.json')
preprocessed_data = preprocess_data(dataset)
tokenizer = create_tokenizer(preprocessed_data['texts'])
sequences = prepare_sequences(preprocessed_data['texts'], tokenizer, preprocessed_data['max_sequence_length'])
print("Dataset loading and preprocessing completed.")

# Build and train the model
input_vocab_size = len(tokenizer.word_index) + 1
target_vocab_size = len(tokenizer.word_index) + 1
max_sequence_length = preprocessed_data['max_sequence_length']
mha_output_size = 256

# Prepare actual encoder input, decoder input, and target output data
encoder_input = prepare_sequences(preprocessed_data['texts'], tokenizer, max_sequence_length)
decoder_input = prepare_sequences(preprocessed_data['titles'], tokenizer, max_sequence_length)
target_output = prepare_sequences(preprocessed_data['titles'], tokenizer, max_sequence_length)

# One-hot encode the target titles
target_output_one_hot = to_categorical(target_output, num_classes=target_vocab_size)

# Train the model
model = build_model(input_vocab_size, target_vocab_size, mha_output_size, max_sequence_length)

# Model save path
save_path = 'title_generation_model.keras'

# Optimizing for M1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# Train the model with early stopping
print("Training started...")
history = model.fit(
    [encoder_input, decoder_input], target_output_one_hot,
    epochs=100, batch_size=1, validation_split=0.2,
    callbacks=[early_stopping]
)
# history = model.fit([encoder_input, decoder_input], target_output_one_hot, epochs=5, batch_size=1, validation_split=0.2)
print("Training completed.")

# Save the model
print("Saving the trained model...")
model.save(save_path)
print("Model saved.")

# Load the trained model
# loaded_model = load_trained_model('trained_model.h5')

# Generate titles using the loaded model
# input_text = "Serverless computing is revolutionizing the way we deploy and manage applications..."
# generated_title = generate_title(loaded_model, input_text)
# print("Generated Title:", generated_title)
