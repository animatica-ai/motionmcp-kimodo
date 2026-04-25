# SPDX-License-Identifier: Apache-2.0
"""KimodoBackbone — wraps a Kimodo SOMA model in the MMCP protocol.

The request must use the model's canonical skeleton (the one served at
``/capabilities``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from kimodo import DEFAULT_MODEL, load_model
from motionmcp import (
    Backbone,
    GenerateRequest,
    ModelSpec,
    MotionResult,
    Skeleton,
)
from motionmcp.errors import ProtocolError
from motionmcp.gltf import matrices_to_quats

from .skeleton import resolve_foot_contact_joints, skeleton_to_mmcp
from .translate import translate_request


class KimodoBackbone(Backbone):
    """Run a Kimodo SOMA model under the MMCP protocol.

    Parameters
    ----------
    model_id
        Kimodo model id (e.g. ``"soma30"``). Defaults to env
        ``KIMODO_MODEL`` if set, otherwise :data:`kimodo.DEFAULT_MODEL`.
    device
        Torch device (e.g. ``"cuda:0"``). Defaults to CUDA if available,
        otherwise CPU.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("KIMODO_MODEL", DEFAULT_MODEL)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self._spec: ModelSpec | None = None
        self._slice_indices: np.ndarray | None = None

    # ----- lifecycle -------------------------------------------------------

    def setup(self) -> None:
        os.environ.setdefault("TEXT_ENCODER_MODE", "dummy")
        print(
            f"[motionmcp-kimodo] loading {self.model_id} on {self.device}",
            flush=True,
        )
        self.model = load_model(self.model_id, device=self.device)

        input_skel = self.model.skeleton
        output_skel = getattr(self.model, "output_skeleton", input_skel)

        # Foot contacts are reported in the OUTPUT frame. Resolve against
        # the input joint list (what we serve as the canonical) so clients
        # only see joints they sent.
        contact_joints = resolve_foot_contact_joints(
            list(input_skel.bone_order_names)
        )

        self._spec = ModelSpec(
            id=self.model_id,
            fps=float(self.model.fps),
            canonical_skeleton=Skeleton.model_validate(skeleton_to_mmcp(input_skel)),
            supports_retargeting=False,
            supported_constraints=["root_path", "effector_target", "pose_keyframe"],
            predicted_contact_joints=contact_joints,
            native_clip_seconds=10.0,
            chunking="none",
            recommended_max_duration_seconds=12.0,
        )

        # Cache the slice indices used to project SOMA77 → SOMA30 (or
        # equivalent) on every request. None when input == output.
        if input_skel is not output_skel:
            output_names = list(output_skel.bone_order_names)
            input_names = list(input_skel.bone_order_names)
            self._slice_indices = np.array(
                [output_names.index(n) for n in input_names], dtype=np.int64,
            )
        else:
            self._slice_indices = None

        print(
            f"[motionmcp-kimodo] ready. fps={self.model.fps} "
            f"canonical_joints={len(input_skel.bone_order_names)}",
            flush=True,
        )

    # ----- protocol --------------------------------------------------------

    def capabilities(self) -> ModelSpec:
        if self._spec is None:
            raise ProtocolError(
                "model_unavailable",
                "model has not finished loading",
            )
        return self._spec

    async def generate(self, req: GenerateRequest) -> MotionResult:
        kwargs = translate_request(req, self.model, self.device)

        try:
            output = self.model(**kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise ProtocolError(
                "resource_exhausted",
                "GPU out of memory; retry with a smaller request",
                details={"device": self.device, "reason": str(exc)},
            ) from exc
        except Exception as exc:
            raise ProtocolError(
                "internal_error",
                f"backbone raised {type(exc).__name__}: {exc}",
            ) from exc

        local_rot_mats = _to_numpy(output["local_rot_mats"])    # (B, T, J_out, 3, 3)
        root_positions = _to_numpy(output["root_positions"])    # (B, T, 3)

        # Slice the wider output skeleton down to the input subset, so the
        # wire response uses only joints the client sent.
        if self._slice_indices is not None:
            local_rot_mats = np.take(local_rot_mats, self._slice_indices, axis=2)

        rotations_quat = matrices_to_quats(local_rot_mats)      # (B, T, J_in, 4)

        foot_contacts: dict[str, np.ndarray] = {}
        contact_joints = self._spec.predicted_contact_joints if self._spec else []
        if contact_joints and "foot_contacts" in output:
            contacts = _to_numpy(output["foot_contacts"]).astype(bool)
            for ch, name in enumerate(contact_joints):
                foot_contacts[name] = contacts[..., ch]

        return MotionResult(
            rotations=rotations_quat.astype(np.float32),
            root_translations=root_positions.astype(np.float32),
            foot_contacts=foot_contacts,
        )


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
