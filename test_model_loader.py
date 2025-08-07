import os
import pytest
from model_loader import load_model

def test_load_default_model():
    model = load_model("mobilenet_head")
    assert model is not None
    # You could also check model.input_shape or model.output_shape if accessible

def test_load_custom_model(tmp_path):
    # create a dummy Keras model file
    fake_model = tmp_path / "fake.keras"
    fake_model.write_text("not a real model")
    with pytest.raises(Exception):
        load_model(str(fake_model))

def test_missing_model_name():
    with pytest.raises(ValueError):
        load_model("nonexistent_model")