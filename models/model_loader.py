import os
from keras import models
from django.conf import settings
from django.core.exceptions import SuspiciousFileOperation
from django.utils._os import safe_join
from .models import Training

def load_latest_model():
    try:
        latest_training = Training.objects.latest('created_at')
        model_path = latest_training.model_file.path

        model_dir = settings.MODEL_FOLDER_PATH
        try:
            safe_model_path = safe_join(model_dir, os.path.basename(model_path))
        except SuspiciousFileOperation:
            raise FileNotFoundError("El archivo del modelo no se encuentra en una ubicación permitida.")

        if not os.path.exists(safe_model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {safe_model_path}")

        model = models.load_model(safe_model_path)
        return model
    except Training.DoesNotExist:
        raise FileNotFoundError("No se encontró ningún modelo entrenado.")
