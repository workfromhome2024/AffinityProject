from django.urls import path
from .views import PredictActionView, RetargetView

urlpatterns = [
    path('api/predict/', PredictActionView.as_view(), name='predict-action'),
    path('api/retarget/', RetargetView.as_view(), name='retarget-action'),
]