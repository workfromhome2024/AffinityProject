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
        # Convert PIL to tensor (C, H, W) float32 [0,1] as expected by the policy
        from torchvision.transforms.functional import to_tensor, resize
        image_tensor = to_tensor(image)
        image_tensor = resize(image_tensor, [256, 256])

        # Build observation dict matching the model's expected image features
        # smolvla_base expects observation.images.camera1/camera2/camera3
        image_features = policy.config.image_features
        observation = {
            "observation.state": torch.zeros(1, 6)  # (batch, state_dim) - replace with real joint states if available
        }
        for key in image_features:
            observation[key] = image_tensor.unsqueeze(0)  # add batch dim (b, c, h, w)

        # 4. Tokenize the instruction for the VLM backbone
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        tokenized = tokenizer(instruction, return_tensors="pt", padding=True)
        observation["observation.language.tokens"] = tokenized["input_ids"]
        observation["observation.language.attention_mask"] = tokenized["attention_mask"].to(dtype=torch.bool)

        # 5. Inference - predict_action_chunk returns (batch, n_action_steps, action_dim)
        with torch.no_grad():
            actions = policy.predict_action_chunk(observation)

        return Response({
            "instruction": instruction,
            "action_chunk": actions[0].tolist()  # Remove batch dim, returns list of action steps
        })