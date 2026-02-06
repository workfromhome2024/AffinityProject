from django.apps import AppConfig
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class SmolvlaConfig(AppConfig):
    name = 'smolvla'
    policy = None

    def ready(self):
        # Prevent double loading in development (RUN_MAIN for dev server),
        # but always load during tests or other non-reloader contexts
        import os
        if os.environ.get('RUN_MAIN') == 'true' or 'RUN_MAIN' not in os.environ:
            print("Loading SmolVLA to CPU...")
            model_id = "lerobot/smolvla_base"  # <- swap checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
            # Set the action chunk size to 50
            self.policy.config.n_action_steps = 50
            self.policy.eval()