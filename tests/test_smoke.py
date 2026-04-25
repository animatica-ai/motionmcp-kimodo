# SPDX-License-Identifier: Apache-2.0
"""Smoke tests that don't require loading model weights.

Anything that needs an actual Kimodo model (FK, generation) is excluded
from CI; those run as integration tests against a GPU-equipped runner.
"""

from __future__ import annotations

import pytest


def test_imports_clean():
    import motionmcp_kimodo
    from motionmcp_kimodo import KimodoBackbone
    from motionmcp_kimodo.cli import main  # noqa: F401
    from motionmcp_kimodo.skeleton import (  # noqa: F401
        resolve_foot_contact_joints,
        skeleton_to_mmcp,
        standing_root_position,
    )
    from motionmcp_kimodo.translate import translate_request  # noqa: F401

    assert motionmcp_kimodo.__version__ == "0.1.0"
    assert KimodoBackbone


def test_resolve_foot_contact_joints_full_set():
    from motionmcp_kimodo.skeleton import resolve_foot_contact_joints

    bones = ["Hips", "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    resolved = resolve_foot_contact_joints(bones)
    assert resolved == ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]


def test_resolve_foot_contact_joints_missing_returns_empty():
    from motionmcp_kimodo.skeleton import resolve_foot_contact_joints

    bones = ["Hips", "LeftFoot"]   # no right side at all
    assert resolve_foot_contact_joints(bones) == []


def test_backbone_does_not_load_until_setup():
    """KimodoBackbone() must not try to load the model on construction —
    setup() is the lifecycle hook for that."""
    from motionmcp_kimodo import KimodoBackbone

    b = KimodoBackbone(model_id="some-model", device="cpu")
    assert b.model is None
    assert b._spec is None


def test_capabilities_before_setup_raises():
    from motionmcp.errors import ProtocolError
    from motionmcp_kimodo import KimodoBackbone

    b = KimodoBackbone(model_id="x", device="cpu")
    with pytest.raises(ProtocolError) as ei:
        b.capabilities()
    assert ei.value.code == "model_unavailable"
