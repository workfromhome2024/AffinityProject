import os
import cv2
import numpy as np
import mediapipe as mp
from dex_retargeting.retargeting_config import RetargetingConfig
from celery import Celery

app = Celery('anyteleop')
app.config_from_object({
    'broker_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
})

# Initialize Dex-Retargeting for Unitree G1 with Inspire DFQ hand
URDF_DIR = "/app/robot/g1_description"
CONFIG_PATH = "/app/robot/g1_description/g1_retargeting_config.yaml"

RetargetingConfig.set_default_urdf_dir(URDF_DIR)
retargeting_config = RetargetingConfig.load_from_file(CONFIG_PATH)
retargeter = retargeting_config.build()

# Non-target joints (legs, waist, arms) need default positions (zeros)
# Use the optimizer's own index mapping to get the correct count
num_fixed = len(retargeter.optimizer.idx_pin2fixed)
default_non_target_qpos = np.zeros(num_fixed)

# Human hand landmark indices matching YAML config:
# Row 0 = origin indices, Row 1 = task indices
# [[0,0,0,0,0], [4,8,12,16,20]]
human_origin_indices = retargeting_config.target_link_human_indices[0]  # [0,0,0,0,0]
human_task_indices = retargeting_config.target_link_human_indices[1]    # [4,8,12,16,20]

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

@app.task(name='anyteleop.process_video')
def process_video(video_path):
    if not os.path.exists(video_path):
        return {'error': f'Video not found: {video_path}'}

    cap = cv2.VideoCapture(video_path)
    all_robot_qpos = []
    all_landmarks_2d = []
    last_landmarks_2d = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            # Collect 2D image landmarks (normalized 0-1) for frontend overlay
            if results.multi_hand_landmarks:
                img_landmarks = results.multi_hand_landmarks[0]
                last_landmarks_2d = [[lm.x, lm.y] for lm in img_landmarks.landmark]
            all_landmarks_2d.append(last_landmarks_2d)

            # --- IMPROVEMENT 2: World Landmarks for Metric Consistency ---
            if results.multi_hand_world_landmarks:
                # Use world_landmarks for metric 3D points (meters), 
                # which aligns with URDF robot scaling.
                hand_landmarks = results.multi_hand_world_landmarks[0]
                
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ])

                # Extract direction vectors from landmarks matching the YAML config
                # ref_value shape: (N, 3) where N = number of finger vectors (5)
                origin_pts = landmarks_array[human_origin_indices]  # (5, 3)
                task_pts = landmarks_array[human_task_indices]      # (5, 3)
                ref_value = task_pts - origin_pts                   # (5, 3) direction vectors

                robot_qpos = retargeter.retarget(ref_value, fixed_qpos=default_non_target_qpos)
                all_robot_qpos.append(robot_qpos.tolist())
            else:
                # --- IMPROVEMENT 3: Sequence Stability ---
                # Instead of None, use the last valid position to prevent robot 'glitching'
                last_qpos = all_robot_qpos[-1] if all_robot_qpos else [0.0] * retargeter.optimizer.robot.dof
                all_robot_qpos.append(last_qpos)

    finally:
        cap.release()

    return {
        'status': 'completed',
        'video_path': video_path,
        'frame_count': len(all_robot_qpos),
        'actions': all_robot_qpos,
        'landmarks_2d': all_landmarks_2d,
    }
