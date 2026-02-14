"""
Whole-body inverse kinematics retargeter for Unitree G1 humanoid robot.

Uses Pinocchio for damped least-squares IK to map MediaPipe Pose landmarks
(33 body landmarks) to the G1 robot's 29 body DOF + 12 hand independent DOF.
"""

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None


# MediaPipe Pose landmark indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

# IK target frame names in the URDF, corresponding MediaPipe indices, and weights.
# head_link is excluded: it's a fixed joint that can only move via 3 waist DOF,
# and its URDF chain position (z≈0 at pelvis) makes it unreachable, causing
# the waist to saturate and distort all arm targets.
IK_TARGETS = [
    # (urdf_frame,              mp_index,        weight)
    ('left_elbow_link',         MP_LEFT_ELBOW,   1.0),
    ('right_elbow_link',        MP_RIGHT_ELBOW,  1.0),
    ('left_wrist_yaw_link',     MP_LEFT_WRIST,   1.5),
    ('right_wrist_yaw_link',    MP_RIGHT_WRIST,  1.5),
    ('left_ankle_roll_link',    MP_LEFT_ANKLE,    0.8),
    ('right_ankle_roll_link',   MP_RIGHT_ANKLE,   0.8),
]

# Body joint names in URDF order (29 DOF)
BODY_JOINT_NAMES = [
    # Left leg (6)
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    # Right leg (6)
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    # Waist (3)
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    # Left arm (7)
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    # Right arm (7)
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

# Left hand independent joint names (6 DOF)
LEFT_HAND_JOINT_NAMES = [
    'L_thumb_proximal_yaw_joint',
    'L_thumb_proximal_pitch_joint',
    'L_index_proximal_joint',
    'L_middle_proximal_joint',
    'L_ring_proximal_joint',
    'L_pinky_proximal_joint',
]

# Right hand independent joint names (6 DOF)
RIGHT_HAND_JOINT_NAMES = [
    'R_thumb_proximal_yaw_joint',
    'R_thumb_proximal_pitch_joint',
    'R_index_proximal_joint',
    'R_middle_proximal_joint',
    'R_ring_proximal_joint',
    'R_pinky_proximal_joint',
]

# Coordinate transform: MediaPipe pose_world_landmarks → Pinocchio/URDF frame
#
# MediaPipe world landmarks (verified empirically from debug output):
#   - Origin: center between hips
#   - x: person's LEFT is positive  (L_shoulder x=+0.20, R_shoulder x=-0.13)
#   - y: DOWN is positive           (nose y=-0.60, ankle y=+0.65)
#   - z: AWAY from camera is negative (nose z=-0.39, forward of hips)
#
# URDF G1 frame (Z-up system, standard robotics):
#   - Origin: pelvis
#   - x: FORWARD is positive
#   - y: LEFT is positive
#   - z: UP is positive
#
# Mapping:
#   urdf_x (forward) = -mp_z (nose z<0 → urdf_x>0, correct)
#   urdf_y (left)    =  mp_x (left=left, same sign)
#   urdf_z (up)      = -mp_y (down→up, negate!)
MP_TO_URDF_ROTATION = np.array([
    [0,  0, -1],   # urdf_x = -mp_z
    [1,  0,  0],   # urdf_y =  mp_x
    [0, -1,  0],   # urdf_z = -mp_y  (MP y is DOWN, URDF z is UP)
], dtype=np.float64)

# Human body reference dimensions (meters, approximate)
HUMAN_SHOULDER_WIDTH = 0.40  # typical shoulder width
HUMAN_HIP_WIDTH = 0.30

# G1 robot reference dimensions from URDF (meters)
# Left shoulder offset from torso: y ≈ 0.10022, z ≈ 0.24778
# Hip offset from pelvis: y ≈ 0.064452, z ≈ -0.1027
G1_SHOULDER_WIDTH = 0.200  # 2 * 0.10022
G1_HIP_HEIGHT = 0.1027     # pelvis to hip


class WholeBodyRetargeter:
    """Maps MediaPipe Pose landmarks to G1 robot joint angles via Pinocchio IK."""

    def __init__(self, urdf_path, damping=1e-2, max_iter=50, tolerance=1e-3):
        if pin is None:
            raise ImportError("pinocchio (pin) package is required for whole-body IK")

        self.damping = damping
        self.max_iter = max_iter
        self.tolerance = tolerance

        # Load URDF into Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Build joint name → Pinocchio joint index mapping
        self._joint_name_to_idx = {}
        for i in range(self.model.njoints):
            name = self.model.names[i]
            if name != 'universe':
                self._joint_name_to_idx[name] = i

        # Build joint name → q index mapping (position in configuration vector)
        self._joint_name_to_q_idx = {}
        for name, jidx in self._joint_name_to_idx.items():
            q_idx = self.model.joints[jidx].idx_q
            self._joint_name_to_q_idx[name] = q_idx

        # Map IK target frame names to Pinocchio frame IDs
        self._target_frame_ids = []
        for frame_name, mp_idx, weight in IK_TARGETS:
            fid = self.model.getFrameId(frame_name)
            if fid < self.model.nframes:
                self._target_frame_ids.append((fid, mp_idx, weight))
            else:
                print(f"Warning: frame '{frame_name}' not found in URDF")

        # Identify mimic joints to exclude from IK solving
        # Mimic joints are hand intermediate/distal joints
        self._mimic_joint_names = set()
        self._mimic_relations = {}  # mimic_name -> (parent_name, multiplier, offset)
        mimic_suffixes = [
            'thumb_intermediate_joint', 'thumb_distal_joint',
            'index_intermediate_joint', 'middle_intermediate_joint',
            'ring_intermediate_joint', 'pinky_intermediate_joint',
        ]
        for prefix in ['L_', 'R_']:
            for suffix in mimic_suffixes:
                self._mimic_joint_names.add(prefix + suffix)

        # Define mimic relationships
        self._mimic_relations = {
            'L_thumb_intermediate_joint': ('L_thumb_proximal_pitch_joint', 1.6, 0),
            'L_thumb_distal_joint': ('L_thumb_proximal_pitch_joint', 2.4, 0),
            'L_index_intermediate_joint': ('L_index_proximal_joint', 1.0, 0),
            'L_middle_intermediate_joint': ('L_middle_proximal_joint', 1.0, 0),
            'L_ring_intermediate_joint': ('L_ring_proximal_joint', 1.0, 0),
            'L_pinky_intermediate_joint': ('L_pinky_proximal_joint', 1.0, 0),
            'R_thumb_intermediate_joint': ('R_thumb_proximal_pitch_joint', 1.6, 0),
            'R_thumb_distal_joint': ('R_thumb_proximal_pitch_joint', 2.4, 0),
            'R_index_intermediate_joint': ('R_index_proximal_joint', 1.0, 0),
            'R_middle_intermediate_joint': ('R_middle_proximal_joint', 1.0, 0),
            'R_ring_intermediate_joint': ('R_ring_proximal_joint', 1.0, 0),
            'R_pinky_intermediate_joint': ('R_pinky_proximal_joint', 1.0, 0),
        }

        # Build ordered list of output joint names (body + independent hand joints)
        self._output_joint_names = []
        self._output_q_indices = []
        for name in BODY_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES:
            if name in self._joint_name_to_q_idx:
                self._output_joint_names.append(name)
                self._output_q_indices.append(self._joint_name_to_q_idx[name])

        # Build indices for IK-active joints (body only, not hands)
        self._ik_v_indices = []  # indices into velocity vector for IK
        for name in BODY_JOINT_NAMES:
            if name in self._joint_name_to_idx:
                jidx = self._joint_name_to_idx[name]
                v_idx = self.model.joints[jidx].idx_v
                nv = self.model.joints[jidx].nv
                for k in range(nv):
                    self._ik_v_indices.append(v_idx + k)
        self._ik_v_indices = np.array(self._ik_v_indices)

        # Natural standing pose as starting point (instead of all-zeros neutral)
        self._q_neutral = pin.neutral(self.model)
        self._set_standing_pose(self._q_neutral)

        # Joint limits
        self._q_lower = self.model.lowerPositionLimit.copy()
        self._q_upper = self.model.upperPositionLimit.copy()

        # Compute robot reference limb lengths from FK in neutral config
        # Used for per-limb scaling (arm vs leg vs torso)
        self._robot_ref = self._compute_robot_reference_lengths()

    def _set_standing_pose(self, q):
        """Set a natural standing pose with arms at sides, elbows slightly bent, wrists straight."""
        standing_joints = {
            # Legs: standing straight (zeros)
            # Waist: upright (zeros)
            # Left arm: slightly away from body, elbow bent, wrist straight
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.3,     # arm slightly outward
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.5,             # slight bend
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            # Right arm: mirrored
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.3,   # arm slightly outward (negative for right)
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.5,            # slight bend
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            # Hands: relaxed open
            'L_thumb_proximal_yaw_joint': 0.1,
            'L_thumb_proximal_pitch_joint': 0.1,
            'L_index_proximal_joint': 0.1,
            'L_middle_proximal_joint': 0.1,
            'L_ring_proximal_joint': 0.1,
            'L_pinky_proximal_joint': 0.1,
            'R_thumb_proximal_yaw_joint': 0.1,
            'R_thumb_proximal_pitch_joint': 0.1,
            'R_index_proximal_joint': 0.1,
            'R_middle_proximal_joint': 0.1,
            'R_ring_proximal_joint': 0.1,
            'R_pinky_proximal_joint': 0.1,
        }
        for name, value in standing_joints.items():
            if name in self._joint_name_to_q_idx:
                q[self._joint_name_to_q_idx[name]] = value

    def _compute_robot_reference_lengths(self):
        """Compute robot limb lengths from FK in neutral (standing) config."""
        pin.forwardKinematics(self.model, self.data, self._q_neutral)
        pin.updateFramePlacements(self.model, self.data)

        def frame_pos(name):
            fid = self.model.getFrameId(name)
            return self.data.oMf[fid].translation.copy()

        # Key positions in neutral standing config
        pelvis = np.zeros(3)  # origin
        l_shoulder = frame_pos('left_shoulder_pitch_link')
        r_shoulder = frame_pos('right_shoulder_pitch_link')
        l_elbow = frame_pos('left_elbow_link')
        r_elbow = frame_pos('right_elbow_link')
        l_wrist = frame_pos('left_wrist_yaw_link')
        r_wrist = frame_pos('right_wrist_yaw_link')
        l_hip = frame_pos('left_hip_pitch_link')
        r_hip = frame_pos('right_hip_pitch_link')
        l_ankle = frame_pos('left_ankle_roll_link')
        r_ankle = frame_pos('right_ankle_roll_link')

        # Robot limb lengths (average left+right)
        shoulder_center = (l_shoulder + r_shoulder) / 2.0
        hip_center = (l_hip + r_hip) / 2.0

        arm_len = (np.linalg.norm(l_wrist - l_shoulder) +
                   np.linalg.norm(r_wrist - r_shoulder)) / 2.0
        leg_len = (np.linalg.norm(l_ankle - l_hip) +
                   np.linalg.norm(r_ankle - r_hip)) / 2.0
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        torso_height = np.linalg.norm(shoulder_center - hip_center)

        ref = {
            'arm_len': arm_len,
            'leg_len': leg_len,
            'shoulder_width': shoulder_width,
            'torso_height': torso_height,
            'l_shoulder': l_shoulder,
            'r_shoulder': r_shoulder,
            'l_hip': l_hip,
            'r_hip': r_hip,
        }
        return ref

    @property
    def joint_names(self):
        """Returns ordered list of output joint names (41: 29 body + 6 left hand + 6 right hand)."""
        return list(self._output_joint_names)

    @property
    def ndof(self):
        """Number of output DOF."""
        return len(self._output_joint_names)

    def _compute_scale_and_offset(self, landmarks_3d):
        """Compute scaling factor and pelvis position from human landmarks."""
        left_shoulder = landmarks_3d[MP_LEFT_SHOULDER]
        right_shoulder = landmarks_3d[MP_RIGHT_SHOULDER]
        left_hip = landmarks_3d[MP_LEFT_HIP]
        right_hip = landmarks_3d[MP_RIGHT_HIP]

        # Human shoulder width
        human_sw = np.linalg.norm(left_shoulder - right_shoulder)
        if human_sw < 0.01:
            human_sw = HUMAN_SHOULDER_WIDTH

        # Scale factor: robot / human
        scale = G1_SHOULDER_WIDTH / human_sw

        # Pelvis position (midpoint of hips)
        pelvis_pos = (left_hip + right_hip) / 2.0

        return scale, pelvis_pos

    def _transform_landmarks(self, landmarks_3d, scale, pelvis_pos):
        """Transform MediaPipe landmarks to URDF frame, centered on pelvis."""
        # Center on pelvis
        centered = landmarks_3d - pelvis_pos
        # Scale to robot proportions
        scaled = centered * scale
        # Rotate to URDF frame
        transformed = (MP_TO_URDF_ROTATION @ scaled.T).T
        return transformed

    def _estimate_hand_curl(self, landmarks_3d, side='right'):
        """Estimate finger curl (0=open, 1=closed) from pose landmarks.

        Uses the distance between wrist and index/pinky fingertip landmarks
        relative to hand size to estimate curl.
        """
        if side == 'left':
            wrist_idx = MP_LEFT_WRIST
            index_idx = MP_LEFT_INDEX
        else:
            wrist_idx = MP_RIGHT_WRIST
            index_idx = MP_RIGHT_INDEX

        wrist = landmarks_3d[wrist_idx]
        index_tip = landmarks_3d[index_idx]

        # Distance from wrist to index tip
        dist = np.linalg.norm(index_tip - wrist)

        # Normalize by forearm length (elbow to wrist) as reference
        elbow_idx = MP_LEFT_ELBOW if side == 'left' else MP_RIGHT_ELBOW
        forearm_len = np.linalg.norm(landmarks_3d[elbow_idx] - wrist)
        if forearm_len < 0.01:
            return 0.5  # default half-curl

        # When fingers are extended, index tip is ~0.6x forearm length from wrist
        # When curled, it's much closer (~0.1x)
        ratio = dist / forearm_len
        curl = np.clip(1.0 - (ratio / 0.6), 0.0, 1.0)
        return curl

    def _set_hand_joints(self, q, landmarks_3d):
        """Set hand joint angles based on heuristic finger curl estimation."""
        for side, joint_names in [('left', LEFT_HAND_JOINT_NAMES), ('right', RIGHT_HAND_JOINT_NAMES)]:
            curl = self._estimate_hand_curl(landmarks_3d, side)

            for jname in joint_names:
                if jname not in self._joint_name_to_q_idx:
                    continue
                q_idx = self._joint_name_to_q_idx[jname]

                # Get joint limits
                jidx = self._joint_name_to_idx[jname]
                lower = self.model.lowerPositionLimit[self.model.joints[jidx].idx_q]
                upper = self.model.upperPositionLimit[self.model.joints[jidx].idx_q]

                # Map curl (0-1) to joint range
                if 'thumb_proximal_yaw' in jname:
                    # Thumb yaw: partial curl
                    q[q_idx] = lower + curl * 0.5 * (upper - lower)
                else:
                    # Other fingers: full curl range
                    q[q_idx] = lower + curl * (upper - lower)

        # Apply mimic joints
        for mimic_name, (parent_name, mult, offset) in self._mimic_relations.items():
            if mimic_name in self._joint_name_to_q_idx and parent_name in self._joint_name_to_q_idx:
                parent_q_idx = self._joint_name_to_q_idx[parent_name]
                mimic_q_idx = self._joint_name_to_q_idx[mimic_name]
                q[mimic_q_idx] = q[parent_q_idx] * mult + offset

    def _compute_limb_scales(self, landmarks_3d):
        """Compute per-limb scale factors from human landmarks vs robot reference."""
        # Human landmarks in MP frame (before rotation)
        l_shoulder = landmarks_3d[MP_LEFT_SHOULDER]
        r_shoulder = landmarks_3d[MP_RIGHT_SHOULDER]
        l_wrist = landmarks_3d[MP_LEFT_WRIST]
        r_wrist = landmarks_3d[MP_RIGHT_WRIST]
        l_hip = landmarks_3d[MP_LEFT_HIP]
        r_hip = landmarks_3d[MP_RIGHT_HIP]
        l_ankle = landmarks_3d[MP_LEFT_ANKLE]
        r_ankle = landmarks_3d[MP_RIGHT_ANKLE]

        # Human limb lengths
        human_arm_len = (np.linalg.norm(l_wrist - l_shoulder) +
                         np.linalg.norm(r_wrist - r_shoulder)) / 2.0
        human_leg_len = (np.linalg.norm(l_ankle - l_hip) +
                         np.linalg.norm(r_ankle - r_hip)) / 2.0
        human_shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

        # Per-limb scales: robot / human
        arm_scale = self._robot_ref['arm_len'] / max(human_arm_len, 0.01)
        leg_scale = self._robot_ref['leg_len'] / max(human_leg_len, 0.01)
        torso_scale = self._robot_ref['shoulder_width'] / max(human_shoulder_width, 0.01)

        return {
            'arm': arm_scale,
            'leg': leg_scale,
            'torso': torso_scale,
        }

    def _transform_target(self, mp_landmark, mp_anchor, robot_anchor, scale, pelvis_pos):
        """Transform a single target: center on anchor, scale, rotate, offset to robot anchor."""
        # Vector from human anchor to target in MP frame
        vec_mp = mp_landmark - mp_anchor
        # Scale by limb ratio
        vec_scaled = vec_mp * scale
        # Rotate to URDF frame
        vec_urdf = MP_TO_URDF_ROTATION @ vec_scaled
        # Place relative to robot anchor
        return robot_anchor + vec_urdf

    def retarget(self, landmarks_3d, visibility=None, prev_q=None):
        """Retarget MediaPipe Pose 3D landmarks to robot joint angles.

        Args:
            landmarks_3d: (33, 3) array of MediaPipe pose world landmarks
            visibility: (33,) array of landmark visibility scores (0-1), or None
            prev_q: Previous joint configuration for continuity, or None

        Returns:
            (ndof,) array of joint angles for the output joints
        """
        landmarks_3d = np.asarray(landmarks_3d, dtype=np.float64)

        # Compute per-limb scales
        limb_scales = self._compute_limb_scales(landmarks_3d)

        # Human anchor positions in MP frame
        pelvis_mp = (landmarks_3d[MP_LEFT_HIP] + landmarks_3d[MP_RIGHT_HIP]) / 2.0
        l_shoulder_mp = landmarks_3d[MP_LEFT_SHOULDER]
        r_shoulder_mp = landmarks_3d[MP_RIGHT_SHOULDER]
        l_hip_mp = landmarks_3d[MP_LEFT_HIP]
        r_hip_mp = landmarks_3d[MP_RIGHT_HIP]

        # Robot anchor positions (from neutral config FK)
        robot_l_shoulder = self._robot_ref['l_shoulder']
        robot_r_shoulder = self._robot_ref['r_shoulder']
        robot_l_hip = self._robot_ref['l_hip']
        robot_r_hip = self._robot_ref['r_hip']

        # Map each target: use limb-specific scale and anchor at the parent joint
        # Arm targets: relative to shoulder, scaled by arm_scale
        # Leg targets: relative to hip, scaled by leg_scale
        _target_config = {
            MP_LEFT_ELBOW:  (l_shoulder_mp, robot_l_shoulder, limb_scales['arm']),
            MP_RIGHT_ELBOW: (r_shoulder_mp, robot_r_shoulder, limb_scales['arm']),
            MP_LEFT_WRIST:  (l_shoulder_mp, robot_l_shoulder, limb_scales['arm']),
            MP_RIGHT_WRIST: (r_shoulder_mp, robot_r_shoulder, limb_scales['arm']),
            MP_LEFT_ANKLE:  (l_hip_mp,      robot_l_hip,      limb_scales['leg']),
            MP_RIGHT_ANKLE: (r_hip_mp,      robot_r_hip,      limb_scales['leg']),
        }

        # Initialize configuration
        if prev_q is not None:
            q = prev_q.copy()
        else:
            q = self._q_neutral.copy()

        # Build IK targets: list of (frame_id, target_position, weight)
        targets = []
        for fid, mp_idx, weight in self._target_frame_ids:
            if visibility is not None and visibility[mp_idx] < 0.5:
                continue
            mp_anchor, robot_anchor, scale = _target_config[mp_idx]
            target_pos = self._transform_target(
                landmarks_3d[mp_idx], mp_anchor, robot_anchor, scale, pelvis_mp
            )
            targets.append((fid, target_pos, weight))

        if len(targets) == 0:
            # No valid targets, return previous or neutral
            return self._extract_output(q)

        # Damped least-squares IK iteration
        nv_ik = len(self._ik_v_indices)
        for iteration in range(self.max_iter):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Build stacked error vector and Jacobian
            errors = []
            jacobians = []
            for fid, target_pos, weight in targets:
                # Current frame position
                current_pos = self.data.oMf[fid].translation

                # Position error, weighted
                err = weight * (target_pos - current_pos)
                errors.append(err)

                # Frame Jacobian (6 x nv), take only translation part (3 x nv)
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J_pos = J_full[:3, :]  # 3 x nv

                # Extract only IK-active columns, weighted
                J_active = weight * J_pos[:, self._ik_v_indices]  # 3 x nv_ik
                jacobians.append(J_active)

            # Stack errors and Jacobians
            e = np.concatenate(errors)  # (3*n_targets,)
            J = np.vstack(jacobians)    # (3*n_targets, nv_ik)

            # Check convergence
            if np.linalg.norm(e) < self.tolerance:
                break

            # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T
            lam2 = self.damping ** 2
            dq_ik = J.T @ np.linalg.solve(JJT + lam2 * np.eye(JJT.shape[0]), e)

            # Build full velocity vector
            dv = np.zeros(self.model.nv)
            dv[self._ik_v_indices] = dq_ik

            # Integrate
            q = pin.integrate(self.model, q, dv)

            # Clamp to joint limits
            for i in range(self.model.nq):
                if self._q_lower[i] < self._q_upper[i]:
                    q[i] = np.clip(q[i], self._q_lower[i], self._q_upper[i])

        # Set hand joints via heuristic
        self._set_hand_joints(q, landmarks_3d)

        return self._extract_output(q)

    def _extract_output(self, q):
        """Extract the output joint angles from the full configuration vector."""
        output = np.zeros(len(self._output_q_indices))
        for i, q_idx in enumerate(self._output_q_indices):
            output[i] = q[q_idx]
        return output

    def get_full_q(self, output_angles):
        """Convert output angles back to a full Pinocchio configuration vector.

        Useful for passing as prev_q to subsequent retarget() calls.
        """
        q = self._q_neutral.copy()
        for i, q_idx in enumerate(self._output_q_indices):
            if i < len(output_angles):
                q[q_idx] = output_angles[i]

        # Apply mimic joints
        for mimic_name, (parent_name, mult, offset) in self._mimic_relations.items():
            if mimic_name in self._joint_name_to_q_idx and parent_name in self._joint_name_to_q_idx:
                parent_q_idx = self._joint_name_to_q_idx[parent_name]
                mimic_q_idx = self._joint_name_to_q_idx[mimic_name]
                q[mimic_q_idx] = q[parent_q_idx] * mult + offset

        return q
