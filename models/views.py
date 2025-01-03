import os
import cv2
import numpy as np
import pandas as pd
import logging
from django.conf import settings
from keras import models
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from .models import Sample, Training, EvaluationResult
from .utils import (
    create_folder,
    read_frames_from_directory,
    normalize_frames,
    save_normalized_frames,
    get_keypoints,
    clear_directory,
    normalize_keypoints
)
from .validation_utils import verify_keypoints_structure
from mediapipe.python.solutions.holistic import Holistic
from .model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoCaptureView(APIView):
    def get(self, request):
        return render(request, 'video_capture.html')
    
class PrediccionCaptureView(APIView):
    def get(self, request):
        return render(request, 'prediccion.html')

# Vista para predecir la acción basada en los keypoints generados
class PredictActionView(APIView):
    def post(self, request):
        try:
            model = models.load_model(settings.MODEL_PATH)
            if model is None:
                return Response(
                    {'error': 'El modelo no está disponible.'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            landmarks = request.data.get('landmarks')
            if not landmarks or not isinstance(landmarks, list) or not all(isinstance(point, dict) for point in landmarks):
                return Response({'error': 'No se encontraron landmarks válidos'}, status=status.HTTP_400_BAD_REQUEST)

            keypoints_sequence = np.array([[point['x'], point['y'], point['z']] for point in landmarks])
            keypoints_sequence = normalize_keypoints(keypoints_sequence)

            try:
                verify_keypoints_structure(keypoints_sequence)
            except ValueError as e:
                logger.error("Error en la estructura de keypoints: %s", e)
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

            if keypoints_sequence.shape[0] < settings.MODEL_FRAMES:
                keypoints_sequence = np.pad(keypoints_sequence, ((0, settings.MODEL_FRAMES - keypoints_sequence.shape[0]), (0, 0)), 'constant')
            elif keypoints_sequence.shape[0] > settings.MODEL_FRAMES:
                keypoints_sequence = keypoints_sequence[:settings.MODEL_FRAMES]

            keypoints_sequence = keypoints_sequence.reshape(1, settings.MODEL_FRAMES, settings.LENGTH_KEYPOINTS)

            prediction = model.predict(keypoints_sequence)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            predicted_sample = Sample.objects.get(id=predicted_class_index)

            EvaluationResult.objects.create(
                sample=predicted_sample,
                prediction=predicted_sample.label,
                confidence=confidence
            )

            return Response({
                'predicted_word': predicted_sample.label,
                'confidence': confidence
            })

        except Exception as e:
            logger.error("Error durante la predicción: %s", e)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CaptureSamplesView(APIView):
    def post(self, request):
        label = request.data.get('label')
        video_file = request.FILES.get('video')
        
        if not label or not video_file:
            return Response({'error': 'El label y el archivo de video son requeridos.'}, status=status.HTTP_400_BAD_REQUEST)

        frames_dir = os.path.join(settings.FRAME_ACTIONS_PATH, label)
        try:
            os.makedirs(frames_dir, exist_ok=True)
        except OSError as e:
            logger.error("Error al crear directorio para los frames: %s", e)
            return Response({'error': 'No se pudo crear el directorio.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        temp_video_path = os.path.join(frames_dir, 'temp_video.webm')
        with open(temp_video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        video_capture = cv2.VideoCapture(temp_video_path)

        frame_count = 0
        success, frame = video_capture.read()

        while success:
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:04}.jpg')
            cv2.imwrite(frame_path, frame)
            success, frame = video_capture.read()
            frame_count += 1
        
        video_capture.release()

        os.remove(temp_video_path)

        # Crear la instancia de Sample
        sample = Sample.objects.create(frames_directory=frames_dir, label=label)
        return Response({'message': 'Muestras capturadas exitosamente.', 'sample_id': sample.id}, status=status.HTTP_201_CREATED)

class NormalizeSamplesView(APIView):
    def post(self, request, sample_id):
        try:
            sample = Sample.objects.get(id=sample_id)
            
            frames = read_frames_from_directory(sample.frames_directory)
            normalized_frames = normalize_frames(frames, target_frame_count=settings.MODEL_FRAMES)

            normalized_dir = f'{sample.frames_directory}'
            clear_directory(normalized_dir)
            save_normalized_frames(normalized_dir, normalized_frames)

            sample.normalized_frames_directory = normalized_dir
            sample.save()

            return Response({'message': 'Muestras normalizadas exitosamente.'}, status=status.HTTP_200_OK)
        except Sample.DoesNotExist:
            return Response({'error': 'Sample no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

class CreateKeypointsView(APIView):
    def post(self, request, sample_id):
        try:
            sample = Sample.objects.get(id=sample_id)
            
            holistic = Holistic(static_image_mode=True)
            keypoints_sequence = get_keypoints(holistic, sample.normalized_frames_directory)
            holistic.close()

            keypoints_file = os.path.join(settings.KEYPOINTS_PATH, f'{sample.label}.h5')
            create_folder(settings.KEYPOINTS_PATH)
            pd.DataFrame(keypoints_sequence).to_hdf(keypoints_file, key='keypoints', mode='w')
              
            sample.keypoints_file = keypoints_file
            sample.save()

            return Response({'message': 'Keypoints generados exitosamente.'}, status=status.HTTP_200_OK)
        except Sample.DoesNotExist:
            return Response({'error': 'Sample no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

class TrainModelView(APIView):
    def post(self, request):
        max_length_frames = settings.MODEL_FRAMES
        output_length = request.data.get('output_length', 10)

        keypoints_files = Sample.objects.filter(keypoints_file__isnull=False)
        if not keypoints_files.exists():
            return Response({'error': 'No hay keypoints generados para entrenar el modelo.'}, status=status.HTTP_400_BAD_REQUEST)

        model = get_model(max_length_frames, output_length)

        create_folder(settings.MODEL_FOLDER_PATH)

        model_file_path = os.path.join(settings.MODEL_FOLDER_PATH, f"actions_{settings.MODEL_FRAMES}.h5")

        model.save(model_file_path)

        training = Training.objects.create(model_file=model_file_path)
        
        return Response({'message': 'Modelo entrenado exitosamente.', 'training_id': training.id}, status=status.HTTP_201_CREATED)