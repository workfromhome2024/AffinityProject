import os
import base64

from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser

class PredictActionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        import torch
        from PIL import Image
        from django.apps import apps
        from torchvision.transforms.functional import to_tensor, resize

        # 1. Get the policy (lazy-loaded on first access)
        policy = apps.get_app_config('smolvla').get_policy()

        # 2. Extract inputs
        instruction = request.data.get('instruction', 'pick up the object')

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
        from django.conf import settings
        from django.core.files.base import ContentFile
        from celery import current_app

        # Accept base64-encoded video in JSON body
        video_base64 = request.data.get('video_base64')
        if video_base64:
            video_bytes = base64.b64decode(video_base64)
            video_name = request.data.get('video_name', 'upload.mp4')
            video_obj = Video(original_name=video_name, size=len(video_bytes))
            video_obj.file.save(video_name, ContentFile(video_bytes), save=True)
        elif request.FILES.get('video'):
            video = request.FILES['video']
            video_obj = Video(original_name=video.name, size=video.size)
            video_obj.file.save(video.name, video, save=True)
        else:
            return Response(
                {"error": "A 'video_base64' field (JSON) or 'video' file (multipart) is required."},
                status=400,
            )

        # Build absolute path that the anyteleop worker can access via shared volume
        video_path = os.path.join(settings.MEDIA_ROOT, video_obj.file.name)

        # Dispatch Celery task to anyteleop worker
        task = current_app.send_task(
            'anyteleop.process_video',
            args=[video_path],
            queue='anyteleop',
        )

        return Response(
            {
                "video_id": str(video_obj.id),
                "task_id": task.id,
                "status": "processing",
            },
            status=202,
        )


class TaskResultView(APIView):
    def get(self, request, task_id):
        from celery.result import AsyncResult

        result = AsyncResult(task_id)

        if result.state == 'PENDING':
            return Response({"status": "processing"})
        elif result.state == 'FAILURE':
            return Response({"status": "failed", "error": str(result.result)}, status=500)
        elif result.state == 'SUCCESS':
            return Response({"status": "completed", **result.result})
        else:
            return Response({"status": result.state})


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


class RobotPhysicalModelView(APIView):
    """Serve URDF and mesh files from the robot description directory."""

    def get(self, request, filepath):
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'anyteleop', 'robot', 'g1_description',
        )
        full_path = os.path.normpath(os.path.join(base_dir, filepath))

        # Path traversal protection
        if not full_path.startswith(os.path.normpath(base_dir)):
            raise Http404

        if not os.path.isfile(full_path):
            raise Http404

        # Determine content type
        ext = os.path.splitext(full_path)[1].lower()
        content_types = {
            '.urdf': 'application/xml',
            '.xml': 'application/xml',
            '.stl': 'application/octet-stream',
            '.dae': 'application/xml',
            '.obj': 'text/plain',
            '.mtl': 'text/plain',
        }
        content_type = content_types.get(ext, 'application/octet-stream')

        response = FileResponse(open(full_path, 'rb'), content_type=content_type)
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response