# SPDX-License-Identifier: Apache-2.0
"""Device-placement tests for translate → Kimodo constraints.

Uses only ``SOMASkeleton30`` (asset pickles, no diffusion weights).  MPS tests
are skipped when Apple GPU is unavailable (e.g. Linux CI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from kimodo.constraints import EndEffectorConstraintSet, create_pairs
from kimodo.motion_rep.conditioning import build_condition_dicts
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30
from motionmcp import schemas

from motionmcp_kimodo.skeleton import skeleton_to_mmcp
from motionmcp_kimodo.translate import (
    _constraint_to_kimodo,
    _place_constraint_tensors,
    translate_request,
)

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)


@dataclass
class StubModel:
    """Minimal stand-in for ``Kimodo`` — only ``fps`` + ``skeleton``."""

    fps: float
    skeleton: Any


def _stub() -> StubModel:
    return StubModel(fps=30.0, skeleton=SOMASkeleton30())


def _wire_skeleton() -> dict[str, Any]:
    return skeleton_to_mmcp(SOMASkeleton30())


def _generate_request(**kwargs: Any) -> schemas.GenerateRequest:
    base = dict(
        protocol_version="0.1",
        model="test",
        skeleton=_wire_skeleton(),
    )
    base.update(kwargs)
    return schemas.GenerateRequest.model_construct(**base)


def _tensor_devices(constraint: Any) -> dict[str, str]:
    names = (
        "frame_indices",
        "pos_indices",
        "rot_indices",
        "global_joints_positions",
        "global_joints_rots",
        "smooth_root_2d",
        "global_root_heading",
        "root_y_pos",
    )
    out: dict[str, str] = {}
    for name in names:
        t = getattr(constraint, name, None)
        if isinstance(t, torch.Tensor):
            out[name] = str(t.device)
    return out


def _build_condition_dicts_on_device(constraint: Any) -> None:
    build_condition_dicts([constraint])


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_translate_request_effector_all_tensors_on_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    model = _stub()
    req = _generate_request(
        segments=[
            schemas.TextSegment(type="text", prompt="walk", duration_frames=60),
            schemas.TextSegment(type="text", prompt="run", duration_frames=60),
        ],
        constraints=[
            schemas.EffectorTargetConstraint(
                type="effector_target",
                joint="LeftHand",
                frames=[10, 70],
                positions=[[0.5, 1.0, 0.5], [0.6, 1.0, 0.6]],
            ),
        ],
    )

    kwargs = translate_request(req, model, device)
    c = kwargs["constraint_lst"][0]
    assert isinstance(c, EndEffectorConstraintSet)

    for key in ("frame_indices", "pos_indices", "rot_indices", "global_joints_positions"):
        assert torch.device(_tensor_devices(c)[key]).type == device

    _build_condition_dicts_on_device(c)

    cropped = c.crop_move(60, 120)
    assert cropped.pos_indices.device.type == device
    _build_condition_dicts_on_device(cropped)

    mr = KimodoMotionRep(model.skeleton, fps=model.fps)
    lengths = torch.tensor([60], device=device)
    om, mm = mr.create_conditions_from_constraints_batched(
        [cropped], lengths, to_normalize=False, device=device,
    )
    assert om.device.type == device
    assert mm.device.type == device


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_translate_request_pose_keyframe_on_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    model = _stub()
    req = _generate_request(
        segments=[schemas.TextSegment(type="text", prompt="a", duration_frames=40)],
        constraints=[
            schemas.PoseKeyframeConstraint(
                type="pose_keyframe",
                frame=5,
                joint_rotations={"LeftHand": [0.0, 0.0, 0.0, 1.0]},
                root_position=[0.0, 1.0, 0.0],
                fill_mode="rest",
            ),
        ],
    )
    c = translate_request(req, model, device)["constraint_lst"][0]
    assert c.global_joints_positions.device.type == device
    assert c.global_joints_rots.device.type == device
    _build_condition_dicts_on_device(c)


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_translate_request_root_path_on_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    model = _stub()
    req = _generate_request(
        segments=[schemas.TextSegment(type="text", prompt="a", duration_frames=30)],
        constraints=[
            schemas.RootPathConstraint(
                type="root_path",
                frames=[0, 15],
                positions_xz=[[0.0, 0.0], [1.0, 1.0]],
            ),
        ],
    )
    c = translate_request(req, model, device)["constraint_lst"][0]
    assert c.smooth_root_2d.device.type == device
    assert c.frame_indices.device.type == device
    _build_condition_dicts_on_device(c)


@mps_only
def test_kimodo_create_pairs_rejects_mixed_mps_cpu() -> None:
    frames = torch.tensor([0, 10], dtype=torch.long, device="mps")
    pos_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(RuntimeError, match="CPU tensor to MPS"):
        create_pairs(frames, pos_idx)


@mps_only
def test_effector_without_place_tensors_fails_on_mps() -> None:
    device = "mps"
    skel = SOMASkeleton30()
    n_keys, j = 2, skel.nbjoints
    frames = torch.tensor([0, 30], dtype=torch.long, device=device)
    pos = torch.randn(n_keys, j, 3, device=device, dtype=torch.float32)
    rots = torch.eye(3, device=device, dtype=torch.float32).expand(n_keys, j, 3, 3).contiguous()
    smooth = pos[:, skel.root_idx, [0, 2]]

    c = EndEffectorConstraintSet(
        skel, frames, pos, rots, smooth, joint_names=["LeftHand"],
    )
    assert c.pos_indices.device.type == "cpu"

    with pytest.raises(RuntimeError, match="CPU tensor to MPS"):
        _build_condition_dicts_on_device(c)

    _place_constraint_tensors(c, device)
    assert c.pos_indices.device.type == "mps"
    assert c.skeleton.neutral_joints.device.type == "cpu"
    _build_condition_dicts_on_device(c)


@mps_only
def test_kimodo_constraint_to_breaks_mps_via_skeleton_float64() -> None:
    """``ConstraintSet.to(device)`` must not be used — it moves float64 skeleton buffers."""
    device = "mps"
    skel = SOMASkeleton30()
    frames = torch.tensor([0], dtype=torch.long, device=device)
    pos = torch.randn(1, skel.nbjoints, 3, device=device, dtype=torch.float32)
    rots = torch.eye(3, device=device, dtype=torch.float32).expand(1, skel.nbjoints, 3, 3).contiguous()
    smooth = pos[:, skel.root_idx, [0, 2]]
    c = EndEffectorConstraintSet(
        skel, frames, pos, rots, smooth, joint_names=["LeftHand"],
    )
    with pytest.raises(TypeError, match="float64"):
        c.to(device)


@mps_only
def test_place_constraint_tensors_matches_translate() -> None:
    device = "mps"
    model = _stub()
    c_proto = schemas.EffectorTargetConstraint(
        type="effector_target",
        joint="LeftHand",
        frames=[5],
        positions=[[0.3, 1.0, 0.2]],
    )
    c = _place_constraint_tensors(
        _constraint_to_kimodo(c_proto, model, device, total_frames=60),
        device,
    )
    assert c.pos_indices.device.type == device
    _build_condition_dicts_on_device(c)
