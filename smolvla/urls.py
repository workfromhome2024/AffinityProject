from django.urls import path
from .views import PredictActionView

urlpatterns = [
    path('api/predict/', PredictActionView.as_view(), name='predict-action'),
]