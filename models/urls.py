from django.urls import path
from .views import (
    CaptureSamplesView,
    NormalizeSamplesView,
    CreateKeypointsView,
    TrainModelView,
    PredictActionView,
    VideoCaptureView,
    PrediccionCaptureView
)

urlpatterns = [
    path('capture-samples/', CaptureSamplesView.as_view(), name='capture_samples'),
    path('normalize-samples/<int:sample_id>/', NormalizeSamplesView.as_view(), name='normalize_samples'),
    path('create-keypoints/<int:sample_id>/', CreateKeypointsView.as_view(), name='create_keypoints'),
    path('train-model/', TrainModelView.as_view(), name='train_model'),
    path('predict-action/', PredictActionView.as_view(), name='predict_action'),
    path('video-capture/', VideoCaptureView.as_view(), name='video_capture'),
    path('prediccion-capture/', PrediccionCaptureView.as_view(), name='prediccion_capture'),
]
