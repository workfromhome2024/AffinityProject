from django.apps import AppConfig
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class SmolvlaConfig(AppConfig):
    name = 'smolvla'
    policy = None

    def ready(self):
        # Prevent double loading in development
        import os
        if os.environ.get('RUN_MAIN') == 'true':
            print("Loading SmolVLA to CPU...")
            model_id = "lerobot/smolvla_base"  # <- swap checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
            self.policy.eval()