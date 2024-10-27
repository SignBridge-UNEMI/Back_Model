import os
from tensorflow.keras.models import load_model
from django.conf import settings
from django.core.exceptions import SuspiciousFileOperation
from django.utils._os import safe_join
from .models import Training

def load_latest_model():
    try:
        # Obteniendo el último modelo entrenado
        latest_training = Training.objects.latest('created_at')
        model_path = latest_training.model_file.path

        # Validando la ruta para asegurar que esté en un directorio permitido
        model_dir = settings.MODEL_FOLDER_PATH
        try:
            safe_model_path = safe_join(model_dir, os.path.basename(model_path))
        except SuspiciousFileOperation:
            raise FileNotFoundError("El archivo del modelo no se encuentra en una ubicación permitida.")

        # Verificando que el archivo realmente existe
        if not os.path.exists(safe_model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {safe_model_path}")

        # Cargando el modelo
        model = load_model(safe_model_path)
        return model
    except Training.DoesNotExist:
        raise FileNotFoundError("No se encontró ningún modelo entrenado.")
