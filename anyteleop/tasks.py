import os
import cv2
import numpy as np
from celery import Celery

app = Celery('anyteleop')
app.config_from_object({
    'broker_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
})

# HybrIK input image size
HYBRIK_INPUT_SIZE = (256, 192)

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
        hybrik_model = builder.build_sppe(cfg.MODEL).to(device).eval()
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
        tgt_robot='g1',
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
        'body_pose': output.pred_theta_mats.numpy(),
        'joints_3d': output.pred_uvd.numpy(),
    }


def retarget_frame(hybrik_output):
    """Retarget HybrIK output to G1 robot joint angles via GMR."""
    return retargeter.retarget(hybrik_output)


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
