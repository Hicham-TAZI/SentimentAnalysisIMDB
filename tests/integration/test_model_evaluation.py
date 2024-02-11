# tests/unit/test_model_building.py
import pytest
from src.model_building import create_model  # Adjust import based on your actual structure

def test_create_model():
    model = create_model()  # Add parameters if your function requires them
    assert model is not None, "Model creation failed."
    # Assuming a basic model structure; adjust assertions as needed
    assert hasattr(model, 'compile'), "Model does not have compile attribute."
