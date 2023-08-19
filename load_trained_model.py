import tensorflow as tf
from model_training import build_model

def load_trained_model(model_path, input_vocab_size, max_sequence_length):
    model = build_model(input_vocab_size, input_vocab_size, max_sequence_length, mha_output_size=256)
    model.load_weights(model_path)
    return model

# Example usage
if __name__ == "__main__":
    model_path = "path/to/your/model"  # Replace with the path to your trained model
    max_sequence_length = 100  # Replace with the max sequence length used during training
    input_vocab_size = 10000  # Replace with the input vocab size used during training
    
    loaded_model = load_trained_model(model_path, input_vocab_size, max_sequence_length)
    loaded_model.summary()
