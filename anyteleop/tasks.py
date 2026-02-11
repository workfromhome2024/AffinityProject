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
num_total_joints = retargeter.optimizer.robot.dof
num_target_joints = len(retargeting_config.target_joint_names)
num_non_target = num_total_joints - num_target_joints
default_non_target_qpos = np.zeros(num_non_target)

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
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            # --- IMPROVEMENT 2: World Landmarks for Metric Consistency ---
            if results.multi_hand_world_landmarks:
                # Use world_landmarks for metric 3D points (meters), 
                # which aligns with URDF robot scaling.
                hand_landmarks = results.multi_hand_world_landmarks[0]
                
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ])

                # Stage 2: Dex-retargeting
                # The config handles the mapping from 21 points to robot joints
                robot_qpos = retargeter.retarget(landmarks_array, non_target_qpos=default_non_target_qpos)
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
    }
