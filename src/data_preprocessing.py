# src/data_preprocessing.py
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(x_data, y_data, max_length=500, padding_type='post'):
    """
    Preprocesses the IMDB data for training.

    Args:
    - x_data (numpy.ndarray): The input features, already tokenized and indexed.
    - y_data (numpy.ndarray): The labels.
    - max_length (int): Maximum length of sequences after padding.
    - padding_type (str): Type of padding to apply ('pre' or 'post').

    Returns:
    - x_padded (numpy.ndarray): Padded sequences of input features.
    - y_data (numpy.ndarray): Unmodified labels.
    """
    # Assuming the sequences might not already be padded to max_length
    x_padded = pad_sequences(x_data, maxlen=max_length, padding=padding_type, truncating=padding_type)
    
    return x_padded, y_data
