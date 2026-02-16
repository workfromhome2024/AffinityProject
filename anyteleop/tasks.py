import os
import cv2
import numpy as np
from celery import Celery

app = Celery('anyteleop')
app.config_from_object({
    'broker_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
})

# HybrIK input image size (H, W) — must match config IMAGE_SIZE
HYBRIK_INPUT_SIZE = (256, 256)

# Lazy-loaded globals (initialized on first task call)
_initialized = False
ort_session = None
hybrik_model = None
retargeter = None
device = None


def _init_models():
    """Lazy-load HybrIK and GMR on first task invocation."""
    global _initialized, ort_session, hybrik_model, retargeter, device
    if _initialized:
        return

    import torch
    device = torch.device('cpu')

    # HybrIK — prefer ONNX, fall back to PyTorch
    HYBRIK_ONNX_PATH = os.environ.get('HYBRIK_ONNX_PATH', 'hybrik.onnx')
    HYBRIK_PTH_PATH = os.environ.get('HYBRIK_PTH_PATH', 'hybrik_model.pth')
    HYBRIK_CONFIG_PATH = os.environ.get('HYBRIK_CONFIG_PATH', 'configs/256x192_adam_all.yaml')

    if os.path.exists(HYBRIK_ONNX_PATH):
        import onnxruntime as ort
        ort_session = ort.InferenceSession(HYBRIK_ONNX_PATH, providers=['CPUExecutionProvider'])
    elif os.path.exists(HYBRIK_PTH_PATH):
        from hybrik.models import builder
        from hybrik.utils.config import update_config

        cfg = update_config(HYBRIK_CONFIG_PATH)
        # Propagate USE_KID from DATASET into MODEL.EXTRA where the model expects it
        if 'USE_KID' in cfg.get('DATASET', {}):
            cfg.MODEL.EXTRA['USE_KID'] = cfg.DATASET.USE_KID
        elif 'USE_KID' not in cfg.MODEL.get('EXTRA', {}):
            cfg.MODEL.EXTRA['USE_KID'] = True
        # Override backbone pretrain path to use mounted models directory
        MODELS_DIR = os.environ.get('HYBRIK_MODELS_DIR', '/app/models')
        cfg.MODEL.HR_PRETRAINED = os.path.join(MODELS_DIR, 'pose_hrnet_w48_256x192.pth')
        cfg.MODEL.PRETRAINED = os.path.join(MODELS_DIR, 'hybrik_model.pth')
        # HybrIK uses relative paths (e.g. ./hybrik/...) that assume cwd is repo root
        prev_cwd = os.getcwd()
        os.chdir('/opt/hybrik')
        try:
            hybrik_model = builder.build_sppe(cfg.MODEL).to(device).eval()
        finally:
            os.chdir(prev_cwd)
        state_dict = torch.load(HYBRIK_PTH_PATH, map_location=device)
        hybrik_model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"No HybrIK model found. Provide either {HYBRIK_ONNX_PATH} or {HYBRIK_PTH_PATH}"
        )

    # GMR retargeter — use our tuned IK config with wider leg separation
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting import params as gmr_params
    from pathlib import Path
    custom_config = Path('/app/ik_configs/smplx_to_g1_tuned.json')
    if custom_config.exists():
        gmr_params.IK_CONFIG_DICT['smplx']['unitree_g1_with_hands'] = custom_config
    retargeter = GeneralMotionRetargeting(
        src_human='smplx',
        tgt_robot='unitree_g1_with_hands',
    )

    _initialized = True


def preprocess_frame(bgr_frame):
    """Preprocess a BGR video frame for HybrIK input.

    Resizes to 256x192, normalizes to ImageNet stats, returns a (1,3,H,W) array.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (HYBRIK_INPUT_SIZE[1], HYBRIK_INPUT_SIZE[0]))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW -> NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img


def get_hybrik_output(image_nchw):
    """Run HybrIK inference on a preprocessed image array."""
    import torch

    if ort_session is not None:
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: image_nchw})
        return {
            'root_transl': outputs[0],
            'body_pose': outputs[1],
            'joints_3d': outputs[2],
        }

    input_tensor = torch.from_numpy(image_nchw).to(device)
    with torch.no_grad():
        output = hybrik_model(input_tensor)
    return {
        'root_transl': output.transl.numpy(),
        'body_pose': output.pred_theta_mat.numpy(),
        'joints_3d': output.pred_uvd_jts.numpy(),
        'joints_3d_global': output.pred_xyz_hybrik.numpy(),
    }


# SMPL-X joint names (first 22 body joints, matching HybrIK-X output order)
SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# GMR qpos layout: [3 root_pos + 4 root_quat + 43 joint DOFs]
# These are the 43 actuated joint names from the MuJoCo G1-with-hands model,
# in the order they appear in the qpos vector (after the 7-value root).
GMR_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
    'left_hand_index_0_joint', 'left_hand_index_1_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
    'right_hand_index_0_joint', 'right_hand_index_1_joint',
    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
]

# SMPL-X kinematic tree: parent index for each of the first 22 body joints.
# -1 means root (pelvis has no parent).
SMPLX_PARENTS = [
    -1,  # 0  pelvis
     0,  # 1  left_hip       ← pelvis
     0,  # 2  right_hip      ← pelvis
     0,  # 3  spine1         ← pelvis
     1,  # 4  left_knee      ← left_hip
     2,  # 5  right_knee     ← right_hip
     3,  # 6  spine2         ← spine1
     4,  # 7  left_ankle     ← left_knee
     5,  # 8  right_ankle    ← right_knee
     6,  # 9  spine3         ← spine2
     7,  # 10 left_foot      ← left_ankle
     8,  # 11 right_foot     ← right_ankle
     9,  # 12 neck           ← spine3
     9,  # 13 left_collar    ← spine3
     9,  # 14 right_collar   ← spine3
    12,  # 15 head           ← neck
    13,  # 16 left_shoulder  ← left_collar
    14,  # 17 right_shoulder ← right_collar
    16,  # 18 left_elbow     ← left_shoulder
    17,  # 19 right_elbow    ← right_shoulder
    18,  # 20 left_wrist     ← left_elbow
    19,  # 21 right_wrist    ← right_elbow
]


def _rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to quaternion in (w, x, y, z) order."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def hybrik_to_human_data(hybrik_output, chain_rotations=True):
    """Convert HybrIK output to GMR's expected human_data format.

    GMR expects: {joint_name: [global_position_3d, global_quaternion_wxyz]}
    Positions and rotations stay in SMPL-X native frame (Y-up).
    GMR's IK config handles the Y-up to Z-up mapping internally.

    If chain_rotations=True, treat pred_theta_mat as LOCAL rotations and
    chain them through the kinematic tree. If False, use them directly as
    global orientations (HybrIK may already output globals).
    """
    # Squeeze batch dimension and reshape
    positions_raw = hybrik_output['joints_3d_global'].squeeze()
    rotmats_raw = hybrik_output['body_pose'].squeeze()

    # Reshape to expected dimensions: positions (N, 3), rotmats (N, 3, 3)
    positions = positions_raw.reshape(-1, 3)
    rotmats = rotmats_raw.reshape(-1, 3, 3)

    num_joints = min(len(SMPLX_JOINT_NAMES), positions.shape[0], rotmats.shape[0])

    if chain_rotations:
        # Chain local rotations into global: global[i] = global[parent] @ local[i]
        final_rots = [None] * num_joints
        for i in range(num_joints):
            parent = SMPLX_PARENTS[i]
            if parent < 0 or final_rots[parent] is None:
                final_rots[i] = rotmats[i]
            else:
                final_rots[i] = final_rots[parent] @ rotmats[i]
    else:
        # Use rotation matrices directly (already global)
        final_rots = [rotmats[i] for i in range(num_joints)]

    human_data = {}
    for i in range(num_joints):
        name = SMPLX_JOINT_NAMES[i]
        pos = positions[i]
        quat = _rotmat_to_quat_wxyz(final_rots[i])
        human_data[name] = [pos, quat]

    return human_data


def retarget_frame(hybrik_output):
    """Retarget HybrIK output to G1 robot joint angles via GMR."""
    human_data = hybrik_to_human_data(hybrik_output, chain_rotations=True)
    return retargeter.retarget(human_data)


@app.task(name='anyteleop.process_video')
def process_video(video_path):
    _init_models()

    if not os.path.exists(video_path):
        return {'error': f'Video not found: {video_path}'}

    cap = cv2.VideoCapture(video_path)
    all_robot_qpos = []
    all_joints_3d = []
    debug_info = None
    frame_idx = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_nchw = preprocess_frame(frame)
            hybrik_out = get_hybrik_output(image_nchw)

            # Store 3D joint positions for frontend visualization
            joints_3d = hybrik_out['joints_3d']
            if joints_3d.ndim >= 2:
                all_joints_3d.append(joints_3d.squeeze().tolist())
            else:
                all_joints_3d.append(joints_3d.tolist())

            # Retarget to robot joint angles
            robot_q = retarget_frame(hybrik_out)
            robot_q_list = robot_q.tolist() if hasattr(robot_q, 'tolist') else list(robot_q)
            # Strip root pos (3) + root quat (4) = first 7 values, keep only joint DOFs
            joint_dofs = robot_q_list[7:] if len(robot_q_list) > 7 else robot_q_list
            all_robot_qpos.append(joint_dofs)

            # Capture debug info for first frame
            if debug_info is None:
                debug_info = {
                    'frame_idx': frame_idx,
                    'root_transl': hybrik_out['root_transl'].squeeze().tolist(),
                    'body_pose_shape': list(hybrik_out['body_pose'].shape),
                    'joints_3d_shape': list(joints_3d.shape),
                    'first_frame_robot_q': robot_q_list,
                }

            frame_idx += 1

    finally:
        cap.release()

    return {
        'status': 'completed',
        'video_path': video_path,
        'frame_count': len(all_robot_qpos),
        'actions': all_robot_qpos,
        'joint_names': GMR_JOINT_NAMES,
        'joints_3d': all_joints_3d,
        'debug_first_frame': debug_info,
    }
