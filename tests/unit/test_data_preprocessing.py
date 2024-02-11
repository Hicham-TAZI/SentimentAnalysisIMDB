# tests/unit/test_data_preprocessing.py
import pytest
from tensorflow.keras.datasets import imdb
from src.data_preprocessing import preprocess_data

def test_preprocess_data_with_imdb_sample():
    vocab_size = 10000
    max_length = 500
    (x_train, y_train), _ = imdb.load_data(num_words=vocab_size)
    x_train_padded, y_train = preprocess_data(x_train[:10], y_train[:10], max_length=max_length)

    assert x_train_padded is not None, "Preprocessing returned None."
    assert len(x_train_padded) == 10, "Processed data length does not match input sample size."
    # This test now assumes preprocess_data is responsible for padding
    assert x_train_padded.shape[1] == max_length, "Processed sequences do not match the specified max length."
