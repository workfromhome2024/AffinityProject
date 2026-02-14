import os
import cv2
import numpy as np
import mediapipe as mp
from wholebody_ik import WholeBodyRetargeter, MP_TO_URDF_ROTATION, IK_TARGETS
from celery import Celery

app = Celery('anyteleop')
app.config_from_object({
    'broker_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
})

# Initialize whole-body retargeter with full G1 URDF
URDF_PATH = "/app/robot/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
retargeter = WholeBodyRetargeter(URDF_PATH, damping=1e-2, max_iter=50, tolerance=1e-3)

# MediaPipe Pose detector (full body, 33 landmarks)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

@app.task(name='anyteleop.process_video')
def process_video(video_path):
    if not os.path.exists(video_path):
        return {'error': f'Video not found: {video_path}'}

    cap = cv2.VideoCapture(video_path)
    all_robot_qpos = []
    all_landmarks_2d = []
    last_landmarks_2d = None
    prev_q = None  # for IK continuity
    debug_info = None  # diagnostics for first frame
    frame_idx = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)

            # Collect 2D image landmarks (normalized 0-1) for frontend overlay
            if results.pose_landmarks:
                last_landmarks_2d = [
                    [lm.x, lm.y] for lm in results.pose_landmarks.landmark
                ]
            all_landmarks_2d.append(last_landmarks_2d)

            # Use 3D world landmarks for IK
            if results.pose_world_landmarks:
                world_lm = results.pose_world_landmarks.landmark

                landmarks_3d = np.array([
                    [lm.x, lm.y, lm.z] for lm in world_lm
                ])
                visibility = np.array([lm.visibility for lm in world_lm])

                output_angles = retargeter.retarget(
                    landmarks_3d, visibility=visibility, prev_q=prev_q
                )

                # Capture debug info for first valid frame
                if debug_info is None:
                    import pinocchio as pin
                    scale, pelvis_pos = retargeter._compute_scale_and_offset(landmarks_3d)
                    transformed = retargeter._transform_landmarks(landmarks_3d, scale, pelvis_pos)

                    # Get robot frame positions after IK
                    q_full = retargeter.get_full_q(output_angles)
                    pin.forwardKinematics(retargeter.model, retargeter.data, q_full)
                    pin.updateFramePlacements(retargeter.model, retargeter.data)

                    landmark_names = {
                        0: 'nose', 11: 'L_shoulder', 12: 'R_shoulder',
                        13: 'L_elbow', 14: 'R_elbow', 15: 'L_wrist', 16: 'R_wrist',
                        23: 'L_hip', 24: 'R_hip', 27: 'L_ankle', 28: 'R_ankle',
                    }
                    raw_lm = {}
                    transformed_lm = {}
                    for idx, name in landmark_names.items():
                        raw_lm[name] = [round(float(v), 4) for v in landmarks_3d[idx]]
                        transformed_lm[name] = [round(float(v), 4) for v in transformed[idx]]

                    robot_frames = {}
                    for fid, mp_idx in retargeter._target_frame_ids:
                        fname = retargeter.model.frames[fid].name
                        pos = retargeter.data.oMf[fid].translation
                        robot_frames[fname] = [round(float(v), 4) for v in pos]

                    debug_info = {
                        'frame_idx': frame_idx,
                        'scale': round(float(scale), 4),
                        'pelvis_pos_mp': [round(float(v), 4) for v in pelvis_pos],
                        'raw_landmarks_mp': raw_lm,
                        'transformed_landmarks_urdf': transformed_lm,
                        'robot_frame_positions': robot_frames,
                        'first_frame_angles': {
                            name: round(float(val), 4)
                            for name, val in zip(retargeter.joint_names, output_angles)
                        },
                    }

                prev_q = retargeter.get_full_q(output_angles)
                all_robot_qpos.append(output_angles.tolist())
            else:
                # No pose detected: repeat last frame for stability
                last_qpos = all_robot_qpos[-1] if all_robot_qpos else [0.0] * retargeter.ndof
                all_robot_qpos.append(last_qpos)

            frame_idx += 1

    finally:
        cap.release()

    joint_names = retargeter.joint_names

    return {
        'status': 'completed',
        'video_path': video_path,
        'frame_count': len(all_robot_qpos),
        'actions': all_robot_qpos,
        'joint_names': joint_names,
        'landmarks_2d': all_landmarks_2d,
        'debug_first_frame': debug_info,
    }
