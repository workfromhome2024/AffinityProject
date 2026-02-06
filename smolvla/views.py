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
        from torchvision.transforms.functional import to_tensor, resize

        # 3. Pre-process images for each camera
        # The model expects one image per camera (e.g. camera1, camera2, camera3).
        # Clients should upload files keyed by camera name (e.g. "camera1", "camera2", "camera3").
        image_features = policy.config.image_features
        observation = {
            "observation.state": torch.zeros(1, 6)  # (batch, state_dim) - replace with real joint states if available
        }

        received_any = False
        for key in image_features:
            # Extract short camera name from key like "observation.images.camera1" -> "camera1"
            camera_name = key.split(".")[-1]
            image_file = request.FILES.get(camera_name)
            if image_file:
                image = Image.open(image_file).convert("RGB")
                image_tensor = to_tensor(image)
                image_tensor = resize(image_tensor, [256, 256])
                observation[key] = image_tensor.unsqueeze(0)  # (b, c, h, w)
                received_any = True
            else:
                # Zero tensor for missing cameras
                observation[key] = torch.zeros(1, 3, 256, 256)

        # Also accept a single "image" field as fallback — applied to all cameras
        if not received_any:
            image_file = request.FILES.get('image')
            if not image_file:
                return Response({"error": "At least one camera image is required. "
                                 "Upload per-camera files (e.g. 'camera1', 'camera2', 'camera3') "
                                 "or a single 'image' for all cameras."}, status=400)
            image = Image.open(image_file).convert("RGB")
            image_tensor = to_tensor(image)
            image_tensor = resize(image_tensor, [256, 256]).unsqueeze(0)
            for key in image_features:
                observation[key] = image_tensor

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