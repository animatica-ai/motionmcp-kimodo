# SPDX-License-Identifier: Apache-2.0
"""Translate MMCP ``GenerateRequest`` objects into Kimodo model call kwargs.

Lifted from the (deprecated) ``kimodo.mmcp.translate`` module. This is the
bridge between the protocol (Pydantic models from ``motionmcp.schemas``)
and the Kimodo model: prompts, durations, constraints, options.
Validation that depends on the loaded model (joint existence, frame
range, prompt length, etc.) happens here.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np
import torch

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    LeftFootConstraintSet,
    LeftHandConstraintSet,
    RightFootConstraintSet,
    RightHandConstraintSet,
    Root2DConstraintSet,
)

_EndEffectorTypes = (
    EndEffectorConstraintSet,
    LeftHandConstraintSet,
    RightHandConstraintSet,
    LeftFootConstraintSet,
    RightFootConstraintSet,
)
from motionmcp import schemas
from motionmcp.errors import ProtocolError

from .skeleton import standing_root_position


# ---- Public entry point ---------------------------------------------------

def translate_request(
    req: schemas.GenerateRequest,
    model: Any,
    device: str,
) -> dict[str, Any]:
    """Convert an MMCP request into kwargs for ``model(...)``.

    Returns a dict ready to splat into the Kimodo model call (prompts,
    num_frames, constraint_lst, plus generation options).
    """
    # Frame-rate handshake. Server-side resampling isn't implemented yet, so
    # we require the request's effective fps to match the model's.
    requested_fps = req.timing.fps if req.timing else float(model.fps)
    if abs(requested_fps - float(model.fps)) > 1e-3:
        raise ProtocolError(
            "invalid_options",
            f"timing.fps={requested_fps} does not match model fps={float(model.fps)}; "
            "server-side resampling is not implemented",
            details={"model_fps": float(model.fps), "requested_fps": requested_fps},
        )

    # Prompts + frame counts per segment (or one synthetic segment for
    # constraints-only requests).
    if req.segments:
        prompts = [_segment_text(s) for s in req.segments]
        num_frames = [s.duration_frames for s in req.segments]
    else:
        prompts = [""]
        num_frames = [req.duration_frames or 0]

    total_frames = sum(num_frames)

    # Validate constraint frame ranges against the resolved timeline.
    for idx, c in enumerate(req.constraints):
        _validate_constraint_frames(c, total_frames, idx)

    # Compile Kimodo constraints and move tensor fields to ``device``.
    # Do not call ``ConstraintSet.to(device)``: it also moves ``skeleton``,
    # whose asset buffers are float64 on CPU and break on MPS.
    constraint_lst = [
        _finalize_constraint(
            _constraint_to_kimodo(c, model, device, total_frames),
            device,
        )
        for c in req.constraints
    ]

    opts = req.options or schemas.Options()
    cfg_kwargs = _resolve_guidance(opts.guidance) if opts.guidance else {}

    return {
        "prompts":               prompts,
        "num_frames":            num_frames,
        "constraint_lst":        constraint_lst,
        "num_denoising_steps":   opts.diffusion_steps,
        "num_samples":           opts.num_samples,
        "multi_prompt":          True,
        "num_transition_frames": opts.transition_frames,
        "post_processing":       opts.post_processing,
        "return_numpy":          True,
        **cfg_kwargs,
    }


# ---- Segments -------------------------------------------------------------

def _segment_text(segment: schemas.Segment) -> str:
    if isinstance(segment, schemas.TextSegment):
        return segment.prompt
    # UnconditionedSegment: empty string drives the model's null-prompt path.
    return ""


# ---- Guidance -------------------------------------------------------------

def _resolve_guidance(g: schemas.Guidance) -> dict[str, Any]:
    if g.type == "nocfg":
        return {"cfg_type": "nocfg"}
    if g.type == "regular":
        return {"cfg_type": "regular", "cfg_weight": float(g.weight[0])}
    return {"cfg_type": "separated",
            "cfg_weight": [float(g.weight[0]), float(g.weight[1])]}


# ---- Constraint dispatch --------------------------------------------------

def _validate_constraint_frames(c: schemas.Constraint, total_frames: int, idx: int) -> None:
    last = total_frames - 1
    if isinstance(c, schemas.PoseKeyframeConstraint):
        frames = [c.frame]
    else:
        frames = list(c.frames)
    bad = [f for f in frames if f < 0 or f > last]
    if bad:
        raise ProtocolError(
            "frame_out_of_range",
            f"constraint #{idx} ({c.type}) references frame(s) {bad} outside [0, {last}]",
            details={
                "constraint_index": idx,
                "out_of_range":     bad,
                "total_frames":     total_frames,
            },
        )


_CONSTRAINT_TENSOR_FIELDS = (
    "frame_indices",
    "pos_indices",
    "rot_indices",
    "global_joints_positions",
    "global_joints_rots",
    "smooth_root_2d",
    "global_root_heading",
    "root_y_pos",
)


def _place_constraint_tensors(constraint: Any, device: str) -> Any:
    """Move constraint tensor fields to ``device``; leave ``skeleton`` unchanged."""
    if not device or str(device) == "cpu":
        return constraint
    for name in _CONSTRAINT_TENSOR_FIELDS:
        t = getattr(constraint, name, None)
        if isinstance(t, torch.Tensor):
            setattr(constraint, name, t.to(device))
    return constraint


def _bind_crop_move_device(constraint: Any, device: str) -> None:
    """Kimodo ``crop_move`` rebuilds EE constraints with CPU ``pos_indices`` / ``rot_indices``."""
    orig = constraint.crop_move

    def crop_move(self: Any, start: int, end: int) -> Any:
        return _place_constraint_tensors(orig(start, end), device)

    constraint.crop_move = MethodType(crop_move, constraint)


def _finalize_constraint(constraint: Any, device: str) -> Any:
    """Place tensors on ``device`` and patch EE ``crop_move`` for multi-prompt segments."""
    constraint = _place_constraint_tensors(constraint, device)
    if isinstance(constraint, _EndEffectorTypes):
        _bind_crop_move_device(constraint, device)
    return constraint


def _constraint_to_kimodo(
    c: schemas.Constraint,
    model: Any,
    device: str,
    total_frames: int,
) -> Any:
    if isinstance(c, schemas.RootPathConstraint):
        return _root_path(c, model, device)
    if isinstance(c, schemas.EffectorTargetConstraint):
        return _effector_target(c, model, device, total_frames)
    if isinstance(c, schemas.PoseKeyframeConstraint):
        return _pose_keyframe(c, model, device, total_frames)
    raise ProtocolError(
        "unsupported_constraint",
        f"constraint type {type(c).__name__!r} is not implemented",
    )


# ---- root_path ------------------------------------------------------------

def _root_path(
    c: schemas.RootPathConstraint,
    model: Any,
    device: str,
) -> Root2DConstraintSet:
    frames = torch.tensor(c.frames, dtype=torch.long, device=device)
    pos_xz = torch.tensor(c.positions_xz, dtype=torch.float32, device=device)

    kwargs: dict[str, Any] = {
        "skeleton":       model.skeleton,
        "frame_indices":  frames,
        "smooth_root_2d": pos_xz,
    }
    if c.heading_radians is not None:
        # MMCP heading: right-handed rotation about +Y, 0 faces +Z. Kimodo
        # expects [cos(θ), sin(θ)] (the Hum/SOMA convention).
        cos_sin = [(math.cos(t), math.sin(t)) for t in c.heading_radians]
        kwargs["global_root_heading"] = torch.tensor(
            cos_sin, dtype=torch.float32, device=device
        )

    return Root2DConstraintSet(**kwargs)


# ---- effector_target ------------------------------------------------------

_EE_GROUP_TO_CLASS = {
    "LeftHand":  LeftHandConstraintSet,
    "RightHand": RightHandConstraintSet,
    "LeftFoot":  LeftFootConstraintSet,
    "RightFoot": RightFootConstraintSet,
}

# Anchor bone for the "natural reach direction" per EE group — the joint at
# the *fixed* end of the chain, where it attaches to the body. The reach
# rotation is computed as the rotation from (anchor→pin_rest) to
# (anchor→target); applying it to the pinned joint's rest orientation gives
# a wrist/ankle pose that points along the reach instead of staying in
# T-pose.
_EE_GROUP_ANCHOR = {
    "LeftHand":  "LeftShoulder",
    "RightHand": "RightShoulder",
    "LeftFoot":  "Hips",
    "RightFoot": "Hips",
}


def _rotation_between_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that rotates unit vector ``u`` onto unit vector ``v``.

    Rodrigues' formula. Falls back to a perpendicular-axis 180° rotation when
    ``u`` and ``v`` are antiparallel (the standard formula divides by zero
    there).
    """
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    dot = float(np.dot(u, v))
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-6:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(u[0]) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = seed - float(np.dot(seed, u)) * u
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        x, y, z = axis
        return np.array([
            [-1 + 2 * x * x, 2 * x * y,       2 * x * z      ],
            [2 * x * y,       -1 + 2 * y * y, 2 * y * z      ],
            [2 * x * z,       2 * y * z,       -1 + 2 * z * z],
        ], dtype=np.float32)
    cross = np.cross(u, v)
    kx = np.array([
        [0.0,       -cross[2],  cross[1]],
        [cross[2],  0.0,        -cross[0]],
        [-cross[1], cross[0],   0.0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + kx + kx @ kx / (1.0 + dot)).astype(np.float32)


def _effector_target(
    c: schemas.EffectorTargetConstraint,
    model: Any,
    device: str,
    total_frames: int,
) -> EndEffectorConstraintSet:
    skeleton = model.skeleton
    bone_names = list(skeleton.bone_order_names)
    if c.joint not in bone_names:
        raise ProtocolError(
            "unknown_joint",
            f"effector_target.joint={c.joint!r} is not in the model's skeleton",
            details={"joint": c.joint, "available_joints": bone_names},
        )

    ee_groups = _map_joints_to_ee_groups([c.joint], skeleton)
    if not ee_groups or ee_groups[0] not in _EE_GROUP_TO_CLASS:
        raise ProtocolError(
            "unknown_joint",
            f"effector_target.joint={c.joint!r} is not in a constrainable chain "
            f"(must be part of the left/right hand or foot chain)",
            details={
                "joint":               c.joint,
                "supported_ee_groups": list(_EE_GROUP_TO_CLASS),
            },
        )
    ee_group = ee_groups[0]
    constraint_cls = _EE_GROUP_TO_CLASS[ee_group]

    if ee_group == "LeftHand":
        chain_names = list(skeleton.left_hand_joint_names)
    elif ee_group == "RightHand":
        chain_names = list(skeleton.right_hand_joint_names)
    elif ee_group == "LeftFoot":
        chain_names = list(skeleton.left_foot_joint_names)
    else:
        chain_names = list(skeleton.right_foot_joint_names)

    target_pos = np.asarray(c.positions, dtype=np.float32)
    frames = torch.tensor(c.frames, dtype=torch.long, device=device)
    n_keys, n_joints = len(c.frames), len(bone_names)
    j_idx = bone_names.index(c.joint)

    raw_neutral = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float32)
    neutral = raw_neutral + standing_root_position(skeleton)[None, :]
    positions = np.tile(neutral[None, :, :], (n_keys, 1, 1))

    if c.rotations is not None:
        pin_global_rots = _quats_to_matrices(np.asarray(c.rotations, dtype=np.float32))
    else:
        anchor_rest = neutral[bone_names.index(_EE_GROUP_ANCHOR[ee_group])]
        rest_reach = neutral[j_idx] - anchor_rest
        rest_reach_norm = np.linalg.norm(rest_reach)
        pin_global_rots = np.tile(np.eye(3, dtype=np.float32), (n_keys, 1, 1))
        if rest_reach_norm > 1e-6:
            rest_dir = rest_reach / rest_reach_norm
            for k in range(n_keys):
                reach = target_pos[k] - anchor_rest
                reach_norm = np.linalg.norm(reach)
                if reach_norm < 1e-6:
                    continue
                pin_global_rots[k] = _rotation_between_vectors(rest_dir, reach / reach_norm)

    rest_pin = neutral[j_idx]
    for chain_name in chain_names:
        ci = bone_names.index(chain_name)
        rel_rest = neutral[ci] - rest_pin
        for k in range(n_keys):
            positions[k, ci] = target_pos[k] + pin_global_rots[k] @ rel_rest

    global_rots = np.tile(np.eye(3, dtype=np.float32), (n_keys, n_joints, 1, 1))
    global_rots[:, j_idx] = pin_global_rots
    global_rots_t = torch.tensor(global_rots, device=device)

    smooth_root_2d = torch.tensor(target_pos[:, [0, 2]], dtype=torch.float32, device=device)

    return constraint_cls(
        skeleton,
        frame_indices=frames,
        global_joints_positions=torch.tensor(positions, device=device),
        global_joints_rots=global_rots_t,
        smooth_root_2d=smooth_root_2d,
    )


# ---- pose_keyframe --------------------------------------------------------

def _pose_keyframe(
    c: schemas.PoseKeyframeConstraint,
    model: Any,
    device: str,
    total_frames: int,
):
    bone_names = list(model.skeleton.bone_order_names)
    n_joints = len(bone_names)

    unknown = [j for j in c.joint_rotations if j not in bone_names]
    if unknown:
        raise ProtocolError(
            "unknown_joint",
            f"pose_keyframe.joint_rotations references unknown joints: {unknown}",
            details={"unknown_joints": unknown, "available_joints": bone_names},
        )

    specified_joints = list(c.joint_rotations.keys())

    local_rots = np.tile(np.eye(3, dtype=np.float32), (n_joints, 1, 1))
    if c.joint_rotations:
        quat_arr = np.asarray(list(c.joint_rotations.values()), dtype=np.float32)
        target_idx = [bone_names.index(j) for j in specified_joints]
        local_rots[target_idx] = _quats_to_matrices(quat_arr)
    local_rots_t = torch.tensor(local_rots[None, ...], device=device)

    if c.root_position is not None:
        root_world = np.asarray(c.root_position, dtype=np.float32)
    else:
        root_world = standing_root_position(model.skeleton)
    root_t = torch.tensor(root_world[None, :], device=device)

    global_rots_t, posed_positions, _ = model.skeleton.fk(local_rots_t, root_t)
    global_rots_t = global_rots_t.to(device=device)
    posed_positions = posed_positions.to(device=device)

    frames = torch.tensor([c.frame], dtype=torch.long, device=device)

    smooth_root_2d = None
    if c.root_position is not None:
        smooth_root_2d = torch.tensor(
            [[c.root_position[0], c.root_position[2]]],
            dtype=torch.float32,
            device=device,
        )

    if c.fill_mode == "rest":
        kwargs: dict[str, Any] = {
            "skeleton":                model.skeleton,
            "frame_indices":           frames,
            "global_joints_positions": posed_positions,
            "global_joints_rots":      global_rots_t,
        }
        if smooth_root_2d is not None:
            kwargs["smooth_root_2d"] = smooth_root_2d
        return FullBodyConstraintSet(**kwargs)

    ee_groups = _map_joints_to_ee_groups(specified_joints, model.skeleton)
    if not ee_groups:
        kwargs = {
            "skeleton":                model.skeleton,
            "frame_indices":           frames,
            "global_joints_positions": posed_positions,
            "global_joints_rots":      global_rots_t,
        }
        if smooth_root_2d is not None:
            kwargs["smooth_root_2d"] = smooth_root_2d
        return FullBodyConstraintSet(**kwargs)

    return EndEffectorConstraintSet(
        skeleton=model.skeleton,
        frame_indices=frames,
        global_joints_positions=posed_positions,
        global_joints_rots=global_rots_t,
        smooth_root_2d=smooth_root_2d,
        joint_names=ee_groups,
    )


def _map_joints_to_ee_groups(joint_names: list[str], skeleton: Any) -> list[str]:
    """Map arbitrary skeleton joints to Kimodo's 5 EE group names.

    A joint anywhere in the left-hand chain maps to ``"LeftHand"``, etc.
    The pelvis maps to ``"Hips"``. Joints outside any chain (spine, neck)
    map to nothing. Order is stable + de-duplicated.
    """
    left_foot  = set(getattr(skeleton, "left_foot_joint_names",  []) or [])
    right_foot = set(getattr(skeleton, "right_foot_joint_names", []) or [])
    left_hand  = set(getattr(skeleton, "left_hand_joint_names",  []) or [])
    right_hand = set(getattr(skeleton, "right_hand_joint_names", []) or [])
    pelvis     = skeleton.bone_order_names[int(skeleton.root_idx)]

    seen: list[str] = []
    for jn in joint_names:
        if jn == pelvis:
            group = "Hips"
        elif jn in left_foot:
            group = "LeftFoot"
        elif jn in right_foot:
            group = "RightFoot"
        elif jn in left_hand:
            group = "LeftHand"
        elif jn in right_hand:
            group = "RightHand"
        else:
            continue
        if group not in seen:
            seen.append(group)
    return seen


# ---- Quaternion → matrix --------------------------------------------------

def _quats_to_matrices(quats: np.ndarray) -> np.ndarray:
    """``(..., 4) [x,y,z,w]`` quaternions to ``(..., 3, 3)`` rotation matrices."""
    x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n = np.where(n == 0, 1.0, n)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m = np.empty(quats.shape[:-1] + (3, 3), dtype=np.float32)
    m[..., 0, 0] = 1 - 2 * (yy + zz)
    m[..., 0, 1] = 2 * (xy - wz)
    m[..., 0, 2] = 2 * (xz + wy)
    m[..., 1, 0] = 2 * (xy + wz)
    m[..., 1, 1] = 1 - 2 * (xx + zz)
    m[..., 1, 2] = 2 * (yz - wx)
    m[..., 2, 0] = 2 * (xz - wy)
    m[..., 2, 1] = 2 * (yz + wx)
    m[..., 2, 2] = 1 - 2 * (xx + yy)
    return m
