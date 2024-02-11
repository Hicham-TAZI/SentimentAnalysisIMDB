# tests/unit/test_model_building.py
import pytest
from src.model_building import create_model

def test_create_lstm_model():
    lstm_model = create_model(model_type='LSTM', vocab_size=10000, embedding_dim=100, max_length=200, units=64, dropout_rate=0.2, learning_rate=0.001)
    assert lstm_model is not None, "Failed to create LSTM model."
    assert any(['LSTM' in str(layer.__class__) for layer in lstm_model.layers]), "LSTM model does not contain LSTM layers."

def test_create_gru_model():
    gru_model = create_model(model_type='GRU', vocab_size=10000, embedding_dim=100, max_length=200, units=64, dropout_rate=0.2, learning_rate=0.001)
    assert gru_model is not None, "Failed to create GRU model."
    assert any(['GRU' in str(layer.__class__) for layer in gru_model.layers]), "GRU model does not contain GRU layers."
