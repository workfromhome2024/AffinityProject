from django.urls import path
from .views import PredictActionView, RetargetView, TaskResultView, RoboChatView, RobotPhysicalModelView

urlpatterns = [
    path('api/predict/', PredictActionView.as_view(), name='predict-action'),
    path('api/retarget/', RetargetView.as_view(), name='retarget-action'),
    path('api/retarget/<str:task_id>/', TaskResultView.as_view(), name='task-result'),
    path('api/robochat/', RoboChatView.as_view(), name='robo-chat'),
    path('api/robot-model/<path:filepath>', RobotPhysicalModelView.as_view(), name='robot-model'),
]