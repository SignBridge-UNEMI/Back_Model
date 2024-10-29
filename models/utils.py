import os
import cv2
import numpy as np
import shutil
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic

# Función para crear una carpeta si no existe
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Función para limpiar un directorio
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Función para leer frames de un directorio
def read_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg'):
            frame = cv2.imread(os.path.join(directory, filename))
            frames.append(frame)
    return frames

# Guardar frames normalizados en un directorio
def save_normalized_frames(directory, frames):
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(directory, f'frame_{i:02}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

# Función para interpolar frames
def interpolate_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames
    
    indices = np.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        interpolated_frame = cv2.addWeighted(frames[lower_idx], 1 - weight, frames[upper_idx], weight, 0)
        interpolated_frames.append(interpolated_frame)
    
    return interpolated_frames

# Normalizar cantidad de frames
def normalize_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        step = current_frame_count / target_frame_count
        indices = np.arange(0, current_frame_count, step).astype(int)[:target_frame_count]
        return [frames[i] for i in indices]
    else:
        return frames

# Detección de manos con MediaPipe
def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

# Detección de poses con MediaPipe
def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

# Dibujar puntos clave en la imagen
def draw_keypoints(image, results):
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Función para obtener keypoints
def get_keypoints(holistic, frame_dir):
    frames = read_frames_from_directory(frame_dir)
    keypoints_sequence = []
    
    for frame in frames:
        results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        keypoints_sequence.append(keypoints)
    
    return keypoints_sequence

# Función para extraer keypoints de los resultados de MediaPipe
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    return keypoints

# Función para insertar secuencia de keypoints en DataFrame
def insert_keypoints_sequence(data, sample_num, keypoints_sequence):
    for idx, keypoints in enumerate(keypoints_sequence):
        row = pd.DataFrame([keypoints], columns=[f'kp_{i}' for i in range(len(keypoints))])
        row['sample'] = sample_num
        row['frame'] = idx
        data = pd.concat([data, row], ignore_index=True)
    return data


def normalize_keypoints(keypoints_sequence, target_length=15, target_dimensions=1662):
    # Redimensionar para que la secuencia tenga la cantidad de frames objetivo
    if keypoints_sequence.shape[0] > target_length:
        indices = np.linspace(0, keypoints_sequence.shape[0] - 1, target_length).astype(int)
        keypoints_sequence = keypoints_sequence[indices]
    elif keypoints_sequence.shape[0] < target_length:
        keypoints_sequence = np.pad(keypoints_sequence, ((0, target_length - keypoints_sequence.shape[0]), (0, 0)), 'constant')
    
    # Asegurar que cada frame tenga el número de componentes requeridos
    if keypoints_sequence.shape[1] < target_dimensions:
        # Expandir la dimensión de cada keypoint para cumplir con el tamaño objetivo
        keypoints_sequence = np.pad(keypoints_sequence, ((0, 0), (0, target_dimensions - keypoints_sequence.shape[1])), 'constant')
    elif keypoints_sequence.shape[1] > target_dimensions:
        keypoints_sequence = keypoints_sequence[:, :target_dimensions]
    
    return keypoints_sequence
