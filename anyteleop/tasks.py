import os
import numpy as np
from celery import Celery

app = Celery('anyteleop')
app.config_from_object({
    'broker_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
})

# Lazy-loaded globals (initialized on first task call)
_initialized = False
gvhmr_model = None
smpl_model = None
retargeter = None
device = None


def _init_models():
    """Lazy-load GVHMR, SMPL body model, and GMR retargeter."""
    global _initialized, gvhmr_model, smpl_model, retargeter, device
    if _initialized:
        return

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- GVHMR model via hmr4d (Hydra config + checkpoint) ---
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

    gvhmr_ckpt = os.environ.get('GVHMR_CKPT_PATH', '/app/models/gvhmr/gvhmr_model.ckpt')
    gvhmr_model = DemoPL.load_from_checkpoint(
        gvhmr_ckpt, strict=False, map_location=device,
    ).eval()

    # --- SMPL body model for forward kinematics ---
    import smplx
    smpl_path = os.environ.get('SMPL_MODEL_PATH', '/app/models/smpl')
    smpl_model = smplx.create(
        smpl_path, model_type='smpl', gender='neutral', batch_size=1,
    ).to(device).eval()

    # --- GMR retargeter (unchanged) ---
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


def run_gvhmr_inference(video_path):
    """Run GVHMR full pipeline on an entire video.

    Steps: YOLO detection -> ViTPose 2D keypoints -> HMR2 ViT features
           -> camera estimation -> GVHMR model -> world-space SMPL params

    Returns:
        dict with numpy arrays:
            body_pose:     (F, 23, 3) axis-angle per body joint
            global_orient: (F, 1, 3)  axis-angle root orientation
            betas:         (F, 10)    shape coefficients
            transl:        (F, 3)     global translation
    """
    import torch
    from hmr4d.utils.preproc.vitpose import run_vitpose
    from hmr4d.utils.preproc.slam import run_slam
    from hmr4d.utils.preproc.detect import run_detect
    from hmr4d.utils.preproc.vitfeat import run_vitfeat
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hmr4d.utils.geo.hmr_cam import estimate_K, convert_K_to_K4

    video_path = str(video_path)
    length, height, width = get_video_lwh(video_path)

    # Estimate camera intrinsics from image dimensions
    K_fullimg = convert_K_to_K4(estimate_K(height, width))
    K_fullimg = K_fullimg.unsqueeze(0).expand(length, -1).to(device)

    # Preprocessing pipeline
    bbx_xys = run_detect(video_path, device)
    vitpose = run_vitpose(video_path, bbx_xys, device)
    vit_features = run_vitfeat(video_path, bbx_xys, device)
    cam_angvel = run_slam(video_path, device)

    # Pack preprocessed data for GVHMR model
    data = {
        'length': length,
        'bbx_xys': bbx_xys.to(device),
        'kp2d': vitpose.to(device),
        'f_imgseq': vit_features.to(device),
        'cam_angvel': cam_angvel.to(device),
        'K_fullimg': K_fullimg,
    }

    with torch.no_grad():
        pred = gvhmr_model.predict(data, static_cam=False)

    # Extract world-coordinate SMPL parameters
    smpl_params = pred['smpl_params_global']
    return {
        'body_pose': smpl_params['body_pose'].cpu().numpy(),
        'global_orient': smpl_params['global_orient'].cpu().numpy(),
        'betas': smpl_params['betas'].cpu().numpy(),
        'transl': smpl_params['transl'].cpu().numpy(),
    }


def smpl_forward_kinematics(smpl_params):
    """Run SMPL body model to get joint positions and global rotation matrices.

    Args:
        smpl_params: dict with body_pose (F,23,3), global_orient (F,1,3),
                     betas (F,10), transl (F,3) -- all numpy, axis-angle

    Returns:
        joints_3d: (F, 22, 3) numpy -- world-space joint positions
        global_rotmats: (F, 22, 3, 3) numpy -- global rotation matrices per joint
    """
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix

    num_frames = smpl_params['body_pose'].shape[0]

    body_pose = torch.from_numpy(smpl_params['body_pose']).float().to(device)
    global_orient = torch.from_numpy(smpl_params['global_orient']).float().to(device)
    betas = torch.from_numpy(smpl_params['betas']).float().to(device)
    transl = torch.from_numpy(smpl_params['transl']).float().to(device)

    # Expand betas if broadcasted as single set of shape params
    if betas.shape[0] == 1 and num_frames > 1:
        betas = betas.expand(num_frames, -1)

    # Run SMPL forward pass for 3D joint positions
    with torch.no_grad():
        smpl_out = smpl_model(
            body_pose=body_pose.reshape(num_frames, -1),
            global_orient=global_orient.reshape(num_frames, -1),
            betas=betas,
            transl=transl,
        )
    joints_3d = smpl_out.joints[:, :22, :].cpu().numpy()

    # Build global rotation matrices by chaining local rotations.
    # Joint 0 = global_orient, joints 1-21 = first 21 of body_pose's 23 joints.
    full_pose_aa = torch.cat([global_orient, body_pose[:, :21, :]], dim=1)  # (F, 22, 3)
    local_rotmats = axis_angle_to_matrix(full_pose_aa)  # (F, 22, 3, 3)

    global_rotmats = torch.zeros_like(local_rotmats)
    for i in range(22):
        parent = SMPLX_PARENTS[i]
        if parent < 0:
            global_rotmats[:, i] = local_rotmats[:, i]
        else:
            global_rotmats[:, i] = torch.bmm(
                global_rotmats[:, parent],
                local_rotmats[:, i],
            )

    return joints_3d, global_rotmats.cpu().numpy()


# SMPL-X joint names (first 22 body joints, matching SMPL output order)
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
     0,  # 1  left_hip       <- pelvis
     0,  # 2  right_hip      <- pelvis
     0,  # 3  spine1         <- pelvis
     1,  # 4  left_knee      <- left_hip
     2,  # 5  right_knee     <- right_hip
     3,  # 6  spine2         <- spine1
     4,  # 7  left_ankle     <- left_knee
     5,  # 8  right_ankle    <- right_knee
     6,  # 9  spine3         <- spine2
     7,  # 10 left_foot      <- left_ankle
     8,  # 11 right_foot     <- right_ankle
     9,  # 12 neck           <- spine3
     9,  # 13 left_collar    <- spine3
     9,  # 14 right_collar   <- spine3
    12,  # 15 head           <- neck
    13,  # 16 left_shoulder  <- left_collar
    14,  # 17 right_shoulder <- right_collar
    16,  # 18 left_elbow     <- left_shoulder
    17,  # 19 right_elbow    <- right_shoulder
    18,  # 20 left_wrist     <- left_elbow
    19,  # 21 right_wrist    <- right_elbow
]


def _rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to quaternion in (w, x, y, z) order."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def smpl_to_human_data(positions, rotmats):
    """Convert pre-computed SMPL joint positions and rotations to GMR human_data.

    Args:
        positions: (22, 3) joint positions for one frame
        rotmats: (22, 3, 3) global rotation matrices for one frame

    Returns:
        dict mapping joint_name -> [position_3d, quaternion_wxyz]
    """
    num_joints = min(len(SMPLX_JOINT_NAMES), positions.shape[0], rotmats.shape[0])
    human_data = {}
    for i in range(num_joints):
        name = SMPLX_JOINT_NAMES[i]
        pos = positions[i]
        quat = _rotmat_to_quat_wxyz(rotmats[i])
        human_data[name] = [pos, quat]
    return human_data


def retarget_frame(positions, rotmats):
    """Retarget one frame of SMPL data to G1 robot joint angles via GMR."""
    human_data = smpl_to_human_data(positions, rotmats)
    return retargeter.retarget(human_data)


# Number of leg DOFs at the start of GMR_JOINT_NAMES (indices 0-11).
NUM_LEG_DOFS = 12

# Wrist joint indices and reasonable clamp limits (radians).
# The IK solver sometimes pushes wrist joints to hardware limits when
# it can't converge, producing abnormal angles. Clamp to +/-60 deg.
WRIST_CLAMP = {
    19: (-1.05, 1.05),   # left_wrist_roll
    20: (-1.05, 1.05),   # left_wrist_pitch
    21: (-1.05, 1.05),   # left_wrist_yaw
    33: (-1.05, 1.05),   # right_wrist_roll
    34: (-1.05, 1.05),   # right_wrist_pitch
    35: (-1.05, 1.05),   # right_wrist_yaw
}


def _postprocess_actions(all_qpos):
    """Zero leg joints, clamp wrists, and apply temporal smoothing."""
    if not all_qpos:
        return all_qpos

    arr = np.array(all_qpos, dtype=np.float64)

    # 1. Zero out all leg DOFs (indices 0-11) -- upper-body-only retargeting
    arr[:, :NUM_LEG_DOFS] = 0.0

    # 2. Clamp wrist joints to prevent abnormal angles from IK failures
    for idx, (lo, hi) in WRIST_CLAMP.items():
        if idx < arr.shape[1]:
            arr[:, idx] = np.clip(arr[:, idx], lo, hi)

    # 3. Temporal smoothing on upper-body joints only (Savitzky-Golay)
    if arr.shape[0] >= 5:
        from scipy.signal import savgol_filter
        for j in range(NUM_LEG_DOFS, arr.shape[1]):
            arr[:, j] = savgol_filter(arr[:, j], window_length=5, polyorder=2)

    return arr.tolist()


@app.task(name='anyteleop.process_video')
def process_video(video_path):
    _init_models()

    if not os.path.exists(video_path):
        return {'error': f'Video not found: {video_path}'}

    # 1. Run GVHMR on entire video -> SMPL parameters
    smpl_params = run_gvhmr_inference(video_path)

    # 2. SMPL forward kinematics -> joint positions + global rotation matrices
    joints_3d, rotmats = smpl_forward_kinematics(smpl_params)

    num_frames = joints_3d.shape[0]
    all_robot_qpos = []
    all_joints_3d = []
    debug_info = None

    # 3. Per-frame: build human_data -> GMR retarget -> collect joint DOFs
    for frame_idx in range(num_frames):
        pos_f = joints_3d[frame_idx]    # (22, 3)
        rot_f = rotmats[frame_idx]      # (22, 3, 3)

        all_joints_3d.append(pos_f.tolist())

        robot_q = retarget_frame(pos_f, rot_f)
        robot_q_list = robot_q.tolist() if hasattr(robot_q, 'tolist') else list(robot_q)
        # Strip root pos (3) + root quat (4) = first 7 values, keep only joint DOFs
        joint_dofs = robot_q_list[7:] if len(robot_q_list) > 7 else robot_q_list
        all_robot_qpos.append(joint_dofs)

        if debug_info is None:
            debug_info = {
                'frame_idx': frame_idx,
                'transl': smpl_params['transl'][frame_idx].tolist(),
                'body_pose_shape': list(smpl_params['body_pose'].shape),
                'joints_3d_shape': list(joints_3d.shape),
                'first_frame_robot_q': robot_q_list,
            }

    # 4. Postprocess (zero legs, clamp wrists, smooth)
    all_robot_qpos = _postprocess_actions(all_robot_qpos)

    return {
        'status': 'completed',
        'video_path': video_path,
        'frame_count': len(all_robot_qpos),
        'actions': all_robot_qpos,
        'joint_names': GMR_JOINT_NAMES,
        'joints_3d': all_joints_3d,
        'debug_first_frame': debug_info,
    }
