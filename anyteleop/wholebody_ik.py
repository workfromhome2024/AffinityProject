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
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
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
    ('left_knee_link',          MP_LEFT_KNEE,     0.4),
    ('right_knee_link',         MP_RIGHT_KNEE,    0.4),
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

        # Build indices for IK-active joints (legs + arms, NOT waist — waist is set directly)
        WAIST_JOINTS = {'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'}
        self._ik_v_indices = []  # indices into velocity vector for IK
        self._ik_q_indices = []  # indices into configuration vector for postural task
        for name in BODY_JOINT_NAMES:
            if name in WAIST_JOINTS:
                continue  # waist is set directly from landmarks, not from IK
            if name in self._joint_name_to_idx:
                jidx = self._joint_name_to_idx[name]
                v_idx = self.model.joints[jidx].idx_v
                q_idx = self.model.joints[jidx].idx_q
                nv = self.model.joints[jidx].nv
                nq = self.model.joints[jidx].nq
                for k in range(nv):
                    self._ik_v_indices.append(v_idx + k)
                for k in range(nq):
                    self._ik_q_indices.append(q_idx + k)
        self._ik_v_indices = np.array(self._ik_v_indices)
        self._ik_q_indices = np.array(self._ik_q_indices)

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
        """Compute robot limb segment lengths from FK in neutral (standing) config.

        Returns segment lengths (upper_arm, forearm, thigh, shin) and full reach
        values so that direction-based retargeting can use the robot's own geometry.
        """
        pin.forwardKinematics(self.model, self.data, self._q_neutral)
        pin.updateFramePlacements(self.model, self.data)

        def frame_pos(name):
            fid = self.model.getFrameId(name)
            return self.data.oMf[fid].translation.copy()

        # Key positions in neutral standing config
        l_shoulder = frame_pos('left_shoulder_pitch_link')
        r_shoulder = frame_pos('right_shoulder_pitch_link')
        l_elbow = frame_pos('left_elbow_link')
        r_elbow = frame_pos('right_elbow_link')
        l_wrist = frame_pos('left_wrist_yaw_link')
        r_wrist = frame_pos('right_wrist_yaw_link')
        l_hip = frame_pos('left_hip_pitch_link')
        r_hip = frame_pos('right_hip_pitch_link')
        l_knee = frame_pos('left_knee_link')
        r_knee = frame_pos('right_knee_link')
        l_ankle = frame_pos('left_ankle_roll_link')
        r_ankle = frame_pos('right_ankle_roll_link')

        # Segment lengths (average left+right)
        upper_arm = (np.linalg.norm(l_elbow - l_shoulder) +
                     np.linalg.norm(r_elbow - r_shoulder)) / 2.0
        forearm = (np.linalg.norm(l_wrist - l_elbow) +
                   np.linalg.norm(r_wrist - r_elbow)) / 2.0
        thigh = (np.linalg.norm(l_knee - l_hip) +
                 np.linalg.norm(r_knee - r_hip)) / 2.0
        shin = (np.linalg.norm(l_ankle - l_knee) +
                np.linalg.norm(r_ankle - r_knee)) / 2.0

        ref = {
            'upper_arm': upper_arm,
            'forearm': forearm,
            'full_arm': upper_arm + forearm,   # max arm reach when fully extended
            'thigh': thigh,
            'shin': shin,
            'full_leg': thigh + shin,           # max leg reach when fully extended
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

    def _estimate_hand_curl(self, landmarks_3d, side='right'):
        """Estimate finger and thumb curl (0=open, 1=closed) from pose landmarks.

        Uses index, pinky, and thumb fingertip distances from the wrist,
        normalized by forearm length as a scale reference.

        Returns:
            (finger_curl, thumb_curl) tuple, each in range [0, 1]
        """
        if side == 'left':
            wrist_idx, elbow_idx = MP_LEFT_WRIST, MP_LEFT_ELBOW
            index_idx, pinky_idx, thumb_idx = MP_LEFT_INDEX, MP_LEFT_PINKY, MP_LEFT_THUMB
        else:
            wrist_idx, elbow_idx = MP_RIGHT_WRIST, MP_RIGHT_ELBOW
            index_idx, pinky_idx, thumb_idx = MP_RIGHT_INDEX, MP_RIGHT_PINKY, MP_RIGHT_THUMB

        wrist = landmarks_3d[wrist_idx]

        # Forearm length as scale reference
        forearm_len = np.linalg.norm(landmarks_3d[elbow_idx] - wrist)
        if forearm_len < 0.01:
            return 0.5, 0.5  # default half-curl

        # Finger curl: average of index and pinky distances
        index_dist = np.linalg.norm(landmarks_3d[index_idx] - wrist)
        pinky_dist = np.linalg.norm(landmarks_3d[pinky_idx] - wrist)
        avg_finger_dist = (index_dist + pinky_dist) / 2.0
        finger_ratio = avg_finger_dist / forearm_len
        # Extended fingers ~0.6x forearm length, curled ~0.1x
        finger_curl = np.clip(1.0 - (finger_ratio / 0.6), 0.0, 1.0)

        # Thumb curl: separate estimation (thumb is shorter, ~0.4x when extended)
        thumb_dist = np.linalg.norm(landmarks_3d[thumb_idx] - wrist)
        thumb_ratio = thumb_dist / forearm_len
        thumb_curl = np.clip(1.0 - (thumb_ratio / 0.4), 0.0, 1.0)

        return finger_curl, thumb_curl

    def _set_hand_joints(self, q, landmarks_3d):
        """Set hand joint angles based on heuristic finger curl estimation."""
        for side, joint_names in [('left', LEFT_HAND_JOINT_NAMES), ('right', RIGHT_HAND_JOINT_NAMES)]:
            finger_curl, thumb_curl = self._estimate_hand_curl(landmarks_3d, side)

            for jname in joint_names:
                if jname not in self._joint_name_to_q_idx:
                    continue
                q_idx = self._joint_name_to_q_idx[jname]

                # Get joint limits
                jidx = self._joint_name_to_idx[jname]
                lower = self.model.lowerPositionLimit[self.model.joints[jidx].idx_q]
                upper = self.model.upperPositionLimit[self.model.joints[jidx].idx_q]

                if 'thumb' in jname:
                    # Thumb joints use thumb_curl, yaw gets partial range
                    scale = 0.5 if 'yaw' in jname else 1.0
                    q[q_idx] = lower + thumb_curl * scale * (upper - lower)
                else:
                    # Index/middle/ring/pinky use finger_curl
                    q[q_idx] = lower + finger_curl * (upper - lower)

        # Apply mimic joints
        for mimic_name, (parent_name, mult, offset) in self._mimic_relations.items():
            if mimic_name in self._joint_name_to_q_idx and parent_name in self._joint_name_to_q_idx:
                parent_q_idx = self._joint_name_to_q_idx[parent_name]
                mimic_q_idx = self._joint_name_to_q_idx[mimic_name]
                q[mimic_q_idx] = q[parent_q_idx] * mult + offset

    def _set_waist_from_landmarks(self, q, landmarks_3d):
        """Set waist yaw/roll/pitch directly from shoulder and hip landmarks.

        Uses a deadzone + damping strategy to keep the torso upright by default.
        MediaPipe landmarks fluctuate, so small deviations are zeroed out and
        larger movements are attenuated. This prevents the robot from hunching
        or tilting when the human is approximately standing straight.
        """
        DEADZONE = 0.15   # ignore rotations below ~8.6 degrees (MP noise)
        GAIN = 0.3        # attenuate remaining signal (only 30% passes through)

        l_shoulder = landmarks_3d[MP_LEFT_SHOULDER]
        r_shoulder = landmarks_3d[MP_RIGHT_SHOULDER]
        l_hip = landmarks_3d[MP_LEFT_HIP]
        r_hip = landmarks_3d[MP_RIGHT_HIP]

        # Shoulder and hip midpoints
        shoulder_mid = (l_shoulder + r_shoulder) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0

        # Waist yaw: rotation around vertical (URDF z-axis)
        shoulder_urdf_x = -(l_shoulder[2] - r_shoulder[2])  # -mp_z difference
        shoulder_urdf_y = l_shoulder[0] - r_shoulder[0]      # mp_x difference
        yaw = np.arctan2(shoulder_urdf_x, shoulder_urdf_y)

        # Waist roll: lateral tilt
        shoulder_height_diff = -(l_shoulder[1] - r_shoulder[1])  # URDF z = -mp_y
        shoulder_lateral = np.linalg.norm([shoulder_urdf_y, shoulder_urdf_x])
        roll = np.arctan2(shoulder_height_diff, max(shoulder_lateral, 0.01))

        # Waist pitch: forward/backward lean
        torso_vec = shoulder_mid - hip_mid
        torso_urdf_x = -torso_vec[2]
        torso_urdf_z = -torso_vec[1]
        pitch = -np.arctan2(torso_urdf_x, max(abs(torso_urdf_z), 0.01))

        # Apply deadzone + damping: zero out small angles (noise),
        # attenuate larger ones to keep torso mostly upright
        def dampen(angle):
            if abs(angle) < DEADZONE:
                return 0.0
            return GAIN * (angle - np.sign(angle) * DEADZONE)

        yaw = dampen(yaw)
        roll = dampen(roll)
        pitch = dampen(pitch)

        # Clamp to joint limits and apply
        for name, value in [('waist_yaw_joint', yaw),
                            ('waist_roll_joint', roll),
                            ('waist_pitch_joint', pitch)]:
            if name in self._joint_name_to_q_idx:
                q_idx = self._joint_name_to_q_idx[name]
                jidx = self._joint_name_to_idx[name]
                lower = self.model.lowerPositionLimit[self.model.joints[jidx].idx_q]
                upper = self.model.upperPositionLimit[self.model.joints[jidx].idx_q]
                q[q_idx] = np.clip(value, lower, upper)

    def _dir(self, vec):
        """Normalize a vector to unit length. Returns zero vector if input is too small."""
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-6 else np.zeros(3)

    def _get_current_anchors(self, q):
        """Get current shoulder/hip positions after waist has been set.

        The neutral-config anchor positions become stale once the waist is rotated.
        This runs FK with the current q to get where the shoulders and hips
        actually are, so vector targets are placed correctly.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        def fp(name):
            return self.data.oMf[self.model.getFrameId(name)].translation.copy()

        return {
            'l_shoulder': fp('left_shoulder_pitch_link'),
            'r_shoulder': fp('right_shoulder_pitch_link'),
            'l_hip': fp('left_hip_pitch_link'),
            'r_hip': fp('right_hip_pitch_link'),
        }

    def _compute_leg_targets(self, hip_mp, knee_mp, ankle_mp, robot_hip, ref):
        """Compute knee and ankle targets using crouch-percentage method.

        Instead of raw direction chaining (which is noisy for legs), compute how
        much the human's leg is extended as a percentage of full leg length.
        Apply that same percentage to the G1's leg length.

        Returns:
            (knee_target, ankle_target) tuple of 3D positions in URDF frame
        """
        # Human leg segment lengths
        human_thigh_len = np.linalg.norm(knee_mp - hip_mp)
        human_shin_len = np.linalg.norm(ankle_mp - knee_mp)
        human_full_leg = human_thigh_len + human_shin_len

        if human_full_leg < 0.01:
            # Fallback: straight down at full extension
            knee_target = robot_hip + np.array([0.0, 0.0, -ref['thigh']])
            ankle_target = robot_hip + np.array([0.0, 0.0, -ref['full_leg']])
            return knee_target, ankle_target

        # Hip-to-ankle vector in URDF frame
        hip_to_ankle_urdf = MP_TO_URDF_ROTATION @ (ankle_mp - hip_mp)

        # Vertical drop (z component, should be negative = downward)
        vertical_drop = abs(hip_to_ankle_urdf[2])

        # Crouch percentage: how extended is the leg vertically?
        # 1.0 = fully extended (standing straight), 0.0 = fully crouched
        crouch_pct = np.clip(vertical_drop / human_full_leg, 0.0, 1.0)

        # Scale horizontal offset: ratio of robot/human leg length
        h_scale = ref['full_leg'] / human_full_leg
        horizontal = hip_to_ankle_urdf[:2]  # xy components in URDF
        robot_horizontal = horizontal * h_scale

        # Ankle target: below hip at crouch_pct * G1_full_leg
        robot_vertical = crouch_pct * ref['full_leg']
        ankle_target = robot_hip.copy()
        ankle_target[0] += robot_horizontal[0]
        ankle_target[1] += robot_horizontal[1]
        ankle_target[2] -= robot_vertical

        # Knee target: interpolate between hip and ankle using thigh/leg ratio
        # Use human's hip-to-knee direction for horizontal placement
        hip_to_knee_urdf = MP_TO_URDF_ROTATION @ (knee_mp - hip_mp)
        knee_horizontal = hip_to_knee_urdf[:2] * h_scale
        knee_vertical_drop = abs(hip_to_knee_urdf[2])
        knee_crouch_pct = np.clip(knee_vertical_drop / human_full_leg, 0.0, 1.0)
        knee_target = robot_hip.copy()
        knee_target[0] += knee_horizontal[0]
        knee_target[1] += knee_horizontal[1]
        knee_target[2] -= knee_crouch_pct * ref['full_leg']

        return knee_target, ankle_target

    def _build_vector_targets(self, landmarks_3d, visibility, anchors):
        """Build IK targets using vector-based retargeting.

        Arms: direction-based chaining (robot segment lengths x human limb directions)
        Legs: crouch-percentage method (vertical drop ratio, not raw directions)

        Args:
            landmarks_3d: Raw MediaPipe landmarks
            visibility: Per-landmark visibility scores
            anchors: Current shoulder/hip positions from FK (after waist is set)
        """
        ref = self._robot_ref

        # Human limb landmarks in MediaPipe frame
        l_shoulder_mp = landmarks_3d[MP_LEFT_SHOULDER]
        r_shoulder_mp = landmarks_3d[MP_RIGHT_SHOULDER]
        l_elbow_mp = landmarks_3d[MP_LEFT_ELBOW]
        r_elbow_mp = landmarks_3d[MP_RIGHT_ELBOW]
        l_wrist_mp = landmarks_3d[MP_LEFT_WRIST]
        r_wrist_mp = landmarks_3d[MP_RIGHT_WRIST]
        l_hip_mp = landmarks_3d[MP_LEFT_HIP]
        r_hip_mp = landmarks_3d[MP_RIGHT_HIP]
        l_knee_mp = landmarks_3d[MP_LEFT_KNEE]
        r_knee_mp = landmarks_3d[MP_RIGHT_KNEE]
        l_ankle_mp = landmarks_3d[MP_LEFT_ANKLE]
        r_ankle_mp = landmarks_3d[MP_RIGHT_ANKLE]

        # Compute human limb directions and rotate to URDF frame
        def mp_dir(start, end):
            """Direction from start to end landmark, rotated to URDF frame."""
            return self._dir(MP_TO_URDF_ROTATION @ (end - start))

        # --- ARMS: direction-based chaining ---
        l_upper_arm_dir = mp_dir(l_shoulder_mp, l_elbow_mp)
        l_forearm_dir = mp_dir(l_elbow_mp, l_wrist_mp)
        r_upper_arm_dir = mp_dir(r_shoulder_mp, r_elbow_mp)
        r_forearm_dir = mp_dir(r_elbow_mp, r_wrist_mp)

        # Use CURRENT anchor positions (after waist rotation)
        l_shoulder = anchors['l_shoulder']
        r_shoulder = anchors['r_shoulder']
        l_hip = anchors['l_hip']
        r_hip = anchors['r_hip']

        # Arms: chain from current robot shoulder
        l_elbow_target = l_shoulder + l_upper_arm_dir * ref['upper_arm']
        l_wrist_target = l_elbow_target + l_forearm_dir * ref['forearm']
        r_elbow_target = r_shoulder + r_upper_arm_dir * ref['upper_arm']
        r_wrist_target = r_elbow_target + r_forearm_dir * ref['forearm']

        # --- LEGS: crouch-percentage method ---
        # Use vertical drop ratio instead of raw direction chaining, so legs
        # stay straight when standing and only bend when actually crouching.
        # Also computes knee targets for smoother leg motion during strides.
        l_knee_target, l_ankle_target = self._compute_leg_targets(
            l_hip_mp, l_knee_mp, l_ankle_mp, l_hip, ref)
        r_knee_target, r_ankle_target = self._compute_leg_targets(
            r_hip_mp, r_knee_mp, r_ankle_mp, r_hip, ref)

        # Map MP indices to computed targets
        target_positions = {
            MP_LEFT_ELBOW: l_elbow_target,
            MP_RIGHT_ELBOW: r_elbow_target,
            MP_LEFT_WRIST: l_wrist_target,
            MP_RIGHT_WRIST: r_wrist_target,
            MP_LEFT_KNEE: l_knee_target,
            MP_RIGHT_KNEE: r_knee_target,
            MP_LEFT_ANKLE: l_ankle_target,
            MP_RIGHT_ANKLE: r_ankle_target,
        }

        # Build final target list with visibility filtering
        targets = []
        for fid, mp_idx, weight in self._target_frame_ids:
            if visibility is not None and visibility[mp_idx] < 0.5:
                continue
            targets.append((fid, target_positions[mp_idx], weight))

        return targets

    def retarget(self, landmarks_3d, visibility=None, prev_q=None):
        """Retarget MediaPipe Pose 3D landmarks to robot joint angles.

        Uses vector-based retargeting with task priorities:
          Priority 1 (Balance): Keep torso pointing up via "global up" constraint
          Priority 2 (Direction): Align limb segments to human's unit vectors
          Priority 3 (Relaxation): Postural task pulling toward default standing pose

        Args:
            landmarks_3d: (33, 3) array of MediaPipe pose world landmarks
            visibility: (33,) array of landmark visibility scores (0-1), or None
            prev_q: Previous joint configuration for continuity, or None

        Returns:
            (ndof,) array of joint angles for the output joints
        """
        landmarks_3d = np.asarray(landmarks_3d, dtype=np.float64)

        # Initialize configuration
        if prev_q is not None:
            q = prev_q.copy()
        else:
            q = self._q_neutral.copy()

        # Set waist orientation directly from landmarks (with deadzone + damping)
        self._set_waist_from_landmarks(q, landmarks_3d)

        # Get CURRENT anchor positions after waist rotation
        anchors = self._get_current_anchors(q)

        # Build vector-based IK targets using current anchors
        targets = self._build_vector_targets(landmarks_3d, visibility, anchors)

        if len(targets) == 0:
            return self._extract_output(q)

        # Task weights
        POSTURE_WEIGHT = 0.15   # postural relaxation (pull toward standing)
        UPRIGHT_WEIGHT = 2.0    # "global up" constraint (keep torso vertical)

        q_neutral_ik = self._q_neutral[self._ik_q_indices]

        # Torso "global up" constraint: keep torso_link's z-axis aligned with world z
        torso_fid = self.model.getFrameId('torso_link')
        has_torso = torso_fid < self.model.nframes
        if not has_torso and not hasattr(self, '_torso_warned'):
            print("Warning: 'torso_link' frame not found in URDF — "
                  "global up constraint disabled. Check URDF frame names.")
            self._torso_warned = True

        # Damped least-squares IK iteration with task priorities
        nv_ik = len(self._ik_v_indices)
        final_iteration = 0
        final_pos_error = float('inf')
        for iteration in range(self.max_iter):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # === Task 1: Limb position targets ===
            errors = []
            jacobians = []
            for fid, target_pos, weight in targets:
                current_pos = self.data.oMf[fid].translation
                err = weight * (target_pos - current_pos)
                errors.append(err)

                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J_active = weight * J_full[:3, self._ik_v_indices]
                jacobians.append(J_active)

            e_task = np.concatenate(errors)
            J_task = np.vstack(jacobians)

            # === Task 2: "Global up" constraint ===
            # Keep torso z-axis (column 2 of rotation matrix) aligned with world [0,0,1]
            # Error = desired_up - current_up (3D vector)
            if has_torso:
                torso_rot = self.data.oMf[torso_fid].rotation  # 3x3 rotation
                current_up = torso_rot[:, 2]  # local z-axis in world frame
                desired_up = np.array([0.0, 0.0, 1.0])
                e_up = UPRIGHT_WEIGHT * (desired_up - current_up)

                # Jacobian for the orientation part (rows 3:6) of the frame Jacobian
                J_torso_full = pin.computeFrameJacobian(
                    self.model, self.data, q, torso_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                # The angular Jacobian maps dq → angular velocity ω
                # d(R[:,2])/dt = ω × R[:,2], so we use the skew-symmetric relation
                # For small ω: Δup ≈ ω × current_up = -[current_up]× ω
                # Jacobian: J_up = -[current_up]× J_angular
                J_angular = J_torso_full[3:6, self._ik_v_indices]
                skew = np.array([
                    [0, -current_up[2], current_up[1]],
                    [current_up[2], 0, -current_up[0]],
                    [-current_up[1], current_up[0], 0],
                ])
                J_up = UPRIGHT_WEIGHT * (-skew @ J_angular)

                e_task = np.concatenate([e_task, e_up])
                J_task = np.vstack([J_task, J_up])

            # === Task 3: Postural relaxation ===
            q_current_ik = q[self._ik_q_indices]
            e_posture = POSTURE_WEIGHT * (q_neutral_ik - q_current_ik)

            # Stack all tasks
            e = np.concatenate([e_task, e_posture])
            J_posture = POSTURE_WEIGHT * np.eye(nv_ik)
            J = np.vstack([J_task, J_posture])

            # Check convergence on position targets only
            pos_error = np.linalg.norm(np.concatenate(errors))
            final_iteration = iteration
            final_pos_error = pos_error
            if pos_error < self.tolerance:
                break

            # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T
            lam2 = self.damping ** 2
            dq_ik = J.T @ np.linalg.solve(JJT + lam2 * np.eye(JJT.shape[0]), e)

            dv = np.zeros(self.model.nv)
            dv[self._ik_v_indices] = dq_ik
            q = pin.integrate(self.model, q, dv)

            # Clamp to joint limits
            for i in range(self.model.nq):
                if self._q_lower[i] < self._q_upper[i]:
                    q[i] = np.clip(q[i], self._q_lower[i], self._q_upper[i])

        # Store convergence info for diagnostics
        self._last_ik_iterations = final_iteration + 1
        self._last_ik_error = final_pos_error

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
