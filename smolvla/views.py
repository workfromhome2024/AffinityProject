import base64

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from django.apps import apps
import torch
from PIL import Image
import io

class PredictActionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        # 1. Get the policy (lazy-loaded on first access)
        policy = apps.get_app_config('smolvla').get_policy()
        
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


class RetargetView(APIView):
    parser_classes = [JSONParser, MultiPartParser]

    def post(self, request):
        from .models import Video
        from django.core.files.base import ContentFile

        # Accept base64-encoded video in JSON body
        video_base64 = request.data.get('video_base64')
        if video_base64:
            video_bytes = base64.b64decode(video_base64)
            video_name = request.data.get('video_name', 'upload.mp4')
            video_obj = Video(original_name=video_name, size=len(video_bytes))
            video_obj.file.save(video_name, ContentFile(video_bytes), save=True)
            return Response(
                {
                    "received_video": True,
                    "video_id": str(video_obj.id),
                    "video_name": video_name,
                    "video_size": len(video_bytes),
                    "video_path": video_obj.file.name,
                },
                status=200,
            )

        # Fallback: accept multipart file upload
        video = request.FILES.get('video')
        if video:
            video_obj = Video(original_name=video.name, size=video.size)
            video_obj.file.save(video.name, video, save=True)
            return Response(
                {
                    "received_video": True,
                    "video_id": str(video_obj.id),
                    "video_name": video.name,
                    "video_size": video.size,
                    "video_path": video_obj.file.name,
                },
                status=200,
            )

        return Response(
            {"error": "A 'video_base64' field (JSON) or 'video' file (multipart) is required."},
            status=400,
        )


class RoboChatView(APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        message = request.data.get('message')
        if not message:
            return Response(
                {"error": "A 'message' field is required."},
                status=400,
            )

        # Echo reply for now — replace with actual model inference later
        reply = f"You said: {message}"

        return Response(
            {"reply": reply},
            status=200,
        )