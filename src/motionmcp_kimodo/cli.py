# SPDX-License-Identifier: Apache-2.0
"""``motionmcp-kimodo`` CLI entry point."""

from __future__ import annotations

import argparse
import os

from motionmcp import serve

from .backbone import KimodoBackbone


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="motionmcp-kimodo",
        description="Run an MMCP server backed by the Kimodo motion model. "
                    "Requires the canonical skeleton served at /capabilities.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default=None,
        help="Kimodo model id (default: env KIMODO_MODEL, then kimodo.DEFAULT_MODEL)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help="BitsAndBytes quant for the Kimodo text encoder when TEXT_ENCODER_MODE=local "
        "(4bit or 8bit). No effect with the default dummy encoder.",
    )

    args = parser.parse_args()

    if args.quantize:
        os.environ["KIMODO_QUANTIZE"] = args.quantize.lower()
 
    serve(
        KimodoBackbone(model_id=args.model, device=args.device),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
