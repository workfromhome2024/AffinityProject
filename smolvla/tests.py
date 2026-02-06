import io
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
        # Get the first camera key from the dataset metadata
        camera_key = dataset.meta.camera_keys[0]
        # Image is a torch tensor (C, H, W) float32 [0,1], convert to PIL
        pil_image = to_pil_image(sample[camera_key])

        # 2. Convert PIL Image to Django Uploaded File
        image_io = io.BytesIO()
        pil_image.save(image_io, format='JPEG')
        image_io.seek(0)

        uploaded_image = SimpleUploadedFile(
            "hf_sample.jpg",
            image_io.read(),
            content_type="image/jpeg"
        )

        # 3. Use the task instruction if available in dataset, else use generic
        instruction = sample.get("task", "stack the blocks")

        data = {
            'instruction': instruction,
            'image': uploaded_image
        }

        # 4. Request
        response = self.client.post(url, data, format='multipart')

        # 5. Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('action_chunk', response.data)
        self.assertEqual(len(response.data['action_chunk']), 50)
        
        print(f"Test Success! Instruction: '{instruction}'")
        print(f"First action step: {response.data['action_chunk'][0]}")