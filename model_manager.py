
import os
from model_utils import load_model

class ModelManager:
    def __init__(self, available_models):
        self.available_models = available_models

    def load_model_by_name(self, model_name):
        model_path = self.available_models.get(model_name)
        if model_path and os.path.exists(model_path):
            return load_model(model_path)
        raise FileNotFoundError(f"Model {model_name} not found.")

    def upload_custom_model(self, uploaded_file):
        temp_path = "temp_model.keras"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        model = load_model(temp_path)
        os.remove(temp_path)
        return model
