from django.apps import AppConfig


class SmolvlaConfig(AppConfig):
    name = 'smolvla'
    _policy = None

    def get_policy(self):
        if self._policy is None:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            print("Loading SmolVLA to CPU...")
            model_id = "lerobot/smolvla_base"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
            self._policy.config.n_action_steps = 50
            self._policy.eval()
        return self._policy