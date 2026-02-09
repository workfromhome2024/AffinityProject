import base64
import io
import json
import os
import random
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms.functional import to_pil_image

class SmolVLARealDataTests(APITestCase):

    def download_and_sample_vla_data(self, repo_id="lerobot/svla_so100_stacking", num_samples=1):
        """Downloads a dataset via LeRobotDataset and returns random samples."""
        try:
            dataset = LeRobotDataset(repo_id=repo_id, download_videos=True)
            indices = random.sample(range(dataset.num_frames), min(num_samples, dataset.num_frames))
            return dataset, [dataset[i] for i in indices]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    def test_predict_with_huggingface_data(self):
        """Test API using a real image from Hugging Face datasets."""
        url = reverse('predict-action')

        # 1. Get real data sample
        dataset, samples = self.download_and_sample_vla_data(num_samples=1)
        if not samples:
            self.fail("Could not retrieve sample from Hugging Face")

        sample = samples[0]

        # 2. Convert dataset camera images to uploaded files keyed by the model's camera names.
        # The model expects camera1/camera2/camera3; the dataset may have different names
        # (e.g. top/wrist). Map dataset cameras to model cameras in order, reusing the
        # last available image for any extra model cameras.
        from django.apps import apps
        policy = apps.get_app_config('smolvla').get_policy()
        model_camera_names = [k.split(".")[-1] for k in policy.config.image_features]
        dataset_camera_keys = dataset.meta.camera_keys

        data = {}
        for i, cam_name in enumerate(model_camera_names):
            # Use the corresponding dataset camera, or the last one if fewer dataset cameras
            ds_key = dataset_camera_keys[min(i, len(dataset_camera_keys) - 1)]
            pil_image = to_pil_image(sample[ds_key])
            image_io = io.BytesIO()
            pil_image.save(image_io, format='JPEG')
            image_io.seek(0)
            data[cam_name] = SimpleUploadedFile(
                f"{cam_name}.jpg",
                image_io.read(),
                content_type="image/jpeg"
            )

        # 3. Use the task instruction if available in dataset, else use generic
        instruction = sample.get("task", "stack the blocks")
        data['instruction'] = instruction

        # 4. Request
        response = self.client.post(url, data, format='multipart')

        # 5. Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('action_chunk', response.data)
        self.assertEqual(len(response.data['action_chunk']), 50)

        print(f"Test Success! Instruction: '{instruction}'")
        print(f"First action step: {response.data['action_chunk'][0]}")


class RetargetViewTests(APITestCase):
    VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'testdata', 'opening-drawer.webm')

    def test_retarget_with_video_base64(self):
        """Test retarget endpoint accepts base64-encoded video in JSON."""
        url = reverse('retarget-action')
        with open(self.VIDEO_PATH, 'rb') as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode('utf-8')
        response = self.client.post(
            url,
            {'video_base64': video_b64, 'video_name': 'opening-drawer.webm'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['received_video'])
        self.assertEqual(response.data['video_name'], 'opening-drawer.webm')
        self.assertEqual(response.data['video_size'], len(video_bytes))

    def test_retarget_with_video_multipart(self):
        """Test retarget endpoint still accepts multipart file upload."""
        url = reverse('retarget-action')
        with open(self.VIDEO_PATH, 'rb') as f:
            video = SimpleUploadedFile('opening-drawer.webm', f.read(), content_type='video/webm')
        response = self.client.post(url, {'video': video}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['received_video'])
        self.assertEqual(response.data['video_name'], 'opening-drawer.webm')
        self.assertGreater(response.data['video_size'], 0)

    def test_retarget_without_video(self):
        """Test retarget endpoint rejects requests without a video."""
        url = reverse('retarget-action')
        response = self.client.post(url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_retarget_empty_multipart_rejected(self):
        """Test retarget endpoint rejects empty multipart requests."""
        url = reverse('retarget-action')
        response = self.client.post(url, {}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)


class RoboChatViewTests(APITestCase):

    def test_send_message_returns_reply(self):
        """Test robochat endpoint returns a reply for a valid message."""
        url = reverse('robo-chat')
        response = self.client.post(url, {'message': 'Hello robot'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('reply', response.data)
        self.assertIn('Hello robot', response.data['reply'])

    def test_empty_message_rejected(self):
        """Test robochat endpoint rejects requests without a message."""
        url = reverse('robo-chat')
        response = self.client.post(url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)