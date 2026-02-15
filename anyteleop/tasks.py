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

    # GMR retargeter
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
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

# Coordinate correction: SMPL-X (Y-up) to GMR (Z-up)
_COORD_CORRECTION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to quaternion in (w, x, y, z) order."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def hybrik_to_human_data(hybrik_output):
    """Convert HybrIK output to GMR's expected human_data format.

    GMR expects: {joint_name: [position_3d, quaternion_wxyz]}
    """
    # Squeeze batch dimension and reshape
    positions_raw = hybrik_output['joints_3d_global'].squeeze()
    rotmats_raw = hybrik_output['body_pose'].squeeze()

    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"positions shape: {positions_raw.shape}, rotmats shape: {rotmats_raw.shape}")

    # Reshape to expected dimensions: positions (N, 3), rotmats (N, 3, 3)
    positions = positions_raw.reshape(-1, 3)
    rotmats = rotmats_raw.reshape(-1, 3, 3)

    logger.warning(f"reshaped positions: {positions.shape}, rotmats: {rotmats.shape}")

    num_joints = min(len(SMPLX_JOINT_NAMES), positions.shape[0], rotmats.shape[0])

    human_data = {}
    for i in range(num_joints):
        name = SMPLX_JOINT_NAMES[i]
        # Apply coordinate correction
        pos = _COORD_CORRECTION @ positions[i]
        rot = _COORD_CORRECTION @ rotmats[i] @ _COORD_CORRECTION.T
        quat = _rotmat_to_quat_wxyz(rot)
        human_data[name] = [pos, quat]

    return human_data


def retarget_frame(hybrik_output):
    """Retarget HybrIK output to G1 robot joint angles via GMR."""
    human_data = hybrik_to_human_data(hybrik_output)
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
            all_robot_qpos.append(robot_q_list)

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

    joint_names = retargeter.joint_names if hasattr(retargeter, 'joint_names') else []

    return {
        'status': 'completed',
        'video_path': video_path,
        'frame_count': len(all_robot_qpos),
        'actions': all_robot_qpos,
        'joint_names': joint_names,
        'joints_3d': all_joints_3d,
        'debug_first_frame': debug_info,
    }
