# SPDX-License-Identifier: Apache-2.0
"""motionmcp-kimodo — MMCP backbone for the Kimodo motion model.

A drop-in :class:`motionmcp.Backbone` that runs a Kimodo SOMA model under
the MMCP protocol. Clients send the canonical skeleton served at
``/capabilities`` verbatim.

Run a server::

    motionmcp-kimodo --port 8000
    # or:
    python -m motionmcp_kimodo --model soma30 --device cuda:0
"""

from .backbone import KimodoBackbone

__all__ = ["KimodoBackbone"]
__version__ = "0.1.0"
