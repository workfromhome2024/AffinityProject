from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.apps import apps
import torch
from PIL import Image
import io

class PredictActionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        # 1. Get the pre-loaded policy
        policy = apps.get_app_config('smolvla').policy
        
        # 2. Extract inputs
        instruction = request.data.get('instruction', 'pick up the object')
        image_file = request.FILES.get('image')
        
        if not image_file:
            return Response({"error": "Image required"}, status=400)

        # 3. Pre-process Image
        image = Image.open(image_file).convert("RGB")
        # SmolVLA standard preprocessing happens inside the policy.select_action helper
        # We simulate the observation dictionary structure required by LeRobot
        observation = {
            "observation.image": image,
            "observation.state": torch.zeros(6) # Replace with real joint states if available
        }

        # 4. Inference
        with torch.no_grad():
            # task=instruction feeds the text prompt to the VLM backbone
            action = policy.select_action(observation, task=instruction)

        return Response({
            "instruction": instruction,
            "action_chunk": action.tolist() # Returns a list of predicted actions
        })