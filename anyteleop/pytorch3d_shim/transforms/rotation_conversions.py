"""Minimal rotation conversion utilities matching the pytorch3d API.

Only implements axis_angle_to_matrix and matrix_to_axis_angle,
which are the functions HybrIK imports from pytorch3d.
"""

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle):
    """Convert rotations given as axis/angle to rotation matrices (Rodrigues).

    Args:
        axis_angle: Tensor of shape (..., 3) where the norm of each vector
                    is the rotation angle and the direction is the axis.

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    # Avoid division by zero
    safe_angle = torch.clamp(angle, min=1e-8)
    axis = axis_angle / safe_angle

    # Components
    cos_a = torch.cos(angle).unsqueeze(-1)  # (..., 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (..., 1, 1)

    # Skew-symmetric matrix K
    kx, ky, kz = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zeros = torch.zeros_like(kx)

    # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
    K = torch.stack([
        torch.cat([zeros, -kz, ky], dim=-1),
        torch.cat([kz, zeros, -kx], dim=-1),
        torch.cat([-ky, kx, zeros], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    # Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    # Broadcast I to batch shape
    I = I.expand_as(K)
    R = I + sin_a * K + (1.0 - cos_a) * (K @ K)

    # For near-zero angles, return identity
    small = (angle < 1e-8).unsqueeze(-1)  # (..., 1, 1)
    R = torch.where(small, I, R)

    return R


def matrix_to_axis_angle(matrix):
    """Convert rotation matrices to axis/angle representation.

    Args:
        matrix: Rotation matrices of shape (..., 3, 3).

    Returns:
        Axis-angle vectors of shape (..., 3).
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix):
    """Convert rotation matrices to quaternions (w, x, y, z).

    Args:
        matrix: Rotation matrices of shape (..., 3, 3).

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) order.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    # Shepperd's method
    trace = m00 + m11 + m22

    q = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0  # s = 4*w
    case0 = trace > 0
    q[case0, 0] = 0.25 * s[case0]
    q[case0, 1] = (m21[case0] - m12[case0]) / s[case0]
    q[case0, 2] = (m02[case0] - m20[case0]) / s[case0]
    q[case0, 3] = (m10[case0] - m01[case0]) / s[case0]

    case1 = (~case0) & (m00 > m11) & (m00 > m22)
    s1 = torch.sqrt(torch.clamp(1.0 + m00[case1] - m11[case1] - m22[case1], min=1e-10)) * 2.0
    q[case1, 0] = (m21[case1] - m12[case1]) / s1
    q[case1, 1] = 0.25 * s1
    q[case1, 2] = (m01[case1] + m10[case1]) / s1
    q[case1, 3] = (m02[case1] + m20[case1]) / s1

    case2 = (~case0) & (~case1) & (m11 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m11[case2] - m00[case2] - m22[case2], min=1e-10)) * 2.0
    q[case2, 0] = (m02[case2] - m20[case2]) / s2
    q[case2, 1] = (m01[case2] + m10[case2]) / s2
    q[case2, 2] = 0.25 * s2
    q[case2, 3] = (m12[case2] + m21[case2]) / s2

    case3 = (~case0) & (~case1) & (~case2)
    s3 = torch.sqrt(torch.clamp(1.0 + m22[case3] - m00[case3] - m11[case3], min=1e-10)) * 2.0
    q[case3, 0] = (m10[case3] - m01[case3]) / s3
    q[case3, 1] = (m02[case3] + m20[case3]) / s3
    q[case3, 2] = (m12[case3] + m21[case3]) / s3
    q[case3, 3] = 0.25 * s3

    # Normalize
    q = F.normalize(q, dim=-1)

    return q.reshape(*batch_shape, 4)


def quaternion_to_axis_angle(quaternion):
    """Convert quaternions (w, x, y, z) to axis-angle.

    Args:
        quaternion: Tensor of shape (..., 4) in (w, x, y, z) order.

    Returns:
        Axis-angle vectors of shape (..., 3).
    """
    # Ensure w >= 0 for consistent angle range
    q = quaternion.clone()
    neg_w = q[..., 0:1] < 0
    q = torch.where(neg_w, -q, q)

    # angle = 2 * arccos(w), axis = xyz / sin(angle/2)
    half_angle = torch.acos(torch.clamp(q[..., 0:1], -1.0, 1.0))
    angle = 2.0 * half_angle
    sin_half = torch.sin(half_angle)

    # For small angles, axis doesn't matter (rotation ≈ 0)
    safe_sin = torch.clamp(sin_half, min=1e-8)
    axis = q[..., 1:4] / safe_sin

    return axis * angle
