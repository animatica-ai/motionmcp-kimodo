# SPDX-License-Identifier: Apache-2.0
"""Kimodo skeleton ↔ MMCP wire format helpers.

Lifted from the (deprecated) ``kimodo.mmcp.capabilities`` module so this
project depends only on ``kimodo`` (the model package) and ``motionmcp``
(the SDK), not on any MMCP module inside kimodo.

Two pieces:

* :func:`skeleton_to_mmcp` — turn a Kimodo skeleton object into the
  MMCP ``Skeleton`` dict shape (joints[], coordinate_system, units).
* :func:`resolve_foot_contact_joints` — pick the joint names this
  skeleton uses for the four contact channels Kimodo emits
  ``[L_heel, L_toe, R_heel, R_toe]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# ---- Foot-contact channel mapping ----------------------------------------

# Kimodo emits 4 contact channels in order [L_heel, L_toe, R_heel, R_toe].
# Skeletons without explicit heel/toe joints (e.g. SOMA30 has only
# LeftFoot/LeftToeBase) fall back to the closest available joint so the
# resolved list always has four distinct entries — or empty if the
# skeleton can't supply any of them.
_FOOT_CONTACT_CANDIDATES: list[tuple[str, list[str]]] = [
    ("LeftHeel",  ["LeftHeel",  "L_Heel",  "LeftFoot",      "left_heel"]),
    ("LeftToe",   ["LeftToe",   "LeftToeBase",   "L_Toe",   "left_toe"]),
    ("RightHeel", ["RightHeel", "R_Heel", "RightFoot",     "right_heel"]),
    ("RightToe",  ["RightToe",  "RightToeBase",  "R_Toe",  "right_toe"]),
]


def resolve_foot_contact_joints(bone_names: list[str]) -> list[str]:
    """Pick the actual joint names this skeleton uses for the four contact channels.

    Returns an empty list if any of the four channels has no match — clients
    can then trust that this model won't emit per-joint contacts on this rig.
    """
    bone_set = set(bone_names)
    resolved: list[str] = []
    for _, candidates in _FOOT_CONTACT_CANDIDATES:
        match = next((c for c in candidates if c in bone_set), None)
        if match is None:
            return []
        resolved.append(match)
    return resolved


# ---- Skeleton conversion --------------------------------------------------

def standing_root_offset(skel: Any) -> float:
    """Y-offset that lifts the skeleton so its lowest joint sits on the floor.

    Kimodo training skeletons are pelvis-centered (root at origin, feet
    around Y=−0.99). Shifting up by this offset yields a ground-anchored
    canonical that clients can import as a stand-able rig.
    """
    neutral = _to_numpy(skel.neutral_joints).astype(np.float32)
    min_y = float(neutral[:, 1].min())
    return max(0.0, -min_y)


def standing_root_position(skel: Any) -> np.ndarray:
    """The root's world position in the canonical standing rest pose."""
    return np.array([0.0, standing_root_offset(skel), 0.0], dtype=np.float32)


def skeleton_to_mmcp(skel: Any) -> dict[str, Any]:
    """Convert a Kimodo skeleton object to the MMCP ``Skeleton`` JSON shape.

    Assumes ``neutral_joints`` are global rest positions; we derive
    parent-local ``rest_translation`` by subtracting each joint's parent
    position. Rest rotations are identity (Kimodo canonical skeletons are
    T-posed at rest with no per-joint orientation).

    The root's ``rest_translation`` is shifted upward so the lowest joint
    in the canonical sits at Y=0 — otherwise clients importing the
    canonical would place the character's feet below the floor, and
    generation without an explicit ``root_position`` would centre the
    pelvis at the origin.
    """
    bone_names: list[str] = list(skel.bone_order_names)
    parents = _to_numpy(skel.joint_parents).astype(np.int64)
    neutral_global = _to_numpy(skel.neutral_joints).astype(np.float32)

    floor_offset_y = standing_root_offset(skel)

    joints = []
    for i, name in enumerate(bone_names):
        parent_idx = int(parents[i])
        if parent_idx < 0:
            local_translation = neutral_global[i].copy()
            local_translation[1] += floor_offset_y
            parent_name: str | None = None
        else:
            local_translation = neutral_global[i] - neutral_global[parent_idx]
            parent_name = bone_names[parent_idx]
        joints.append({
            "name":             name,
            "parent":           parent_name,
            "rest_translation": [float(x) for x in local_translation],
            "rest_rotation":    [0.0, 0.0, 0.0, 1.0],
        })

    return {
        "joints":            joints,
        "coordinate_system": "right_handed_y_up",
        "units":             "meters",
    }


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
