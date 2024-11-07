import numpy as np
import os
from django.conf import settings

def verify_keypoints_structure(keypoints):
    if len(keypoints) != settings.MODEL_FRAMES:
        raise ValueError(f"Los keypoints deben tener {settings.MODEL_FRAMES} frames, pero tiene {len(keypoints)}.")
    
    for frame in keypoints:
        if len(frame) != settings.LENGTH_KEYPOINTS:
            raise ValueError(f"Cada frame debe tener {settings.LENGTH_KEYPOINTS} puntos clave, pero tiene {len(frame)}.")

def verify_all_keypoints_files(keypoints_dir):
    for file_name in os.listdir(keypoints_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(keypoints_dir, file_name)
            keypoints = np.load(file_path)
            try:
                verify_keypoints_structure(keypoints)
            except ValueError as e:
                print(f"Error en {file_name}: {e}")
