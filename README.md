# motionmcp-kimodo

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/animatica-ai/motionmcp-kimodo)

**Self-hosted [MMCP](https://animatica.ai/mmcp) server for [Kimodo](https://github.com/animatica-ai/kimodo) motion generation.**

Run Kimodo on your own GPU and expose it over HTTP so any MMCP client can generate animation. The reference client is **[Proscenium for Blender](https://github.com/animatica-ai/proscenium-blender)** — text prompts, paths, and pose constraints on your armature.

> **Alpha** — APIs and packaging may change. Tracks `motionmcp` 0.1.x and Kimodo 1.x.

## Features

- **MMCP-native** — implements [`motionmcp.Backbone`](https://animatica.ai/mmcp/docs/sdk/backbone); capabilities, `/generate`, glTF responses
- **Kimodo SOMA models** — loads Kimodo checkpoints; maps MMCP requests to Kimodo inference
- **Constraint-aware** — root paths, effector targets, pose keyframes (see [MMCP concepts](https://animatica.ai/mmcp/docs/concepts/skeleton))
- **Simple CLI** — `motionmcp-kimodo --port 8000` for local or LAN use
- **Embeddable** — mount `KimodoBackbone` in your own FastAPI / ASGI app ([Development](docs/DEVELOPMENT.md))

## Requirements

| | |
|---|---|
| **Python** | 3.10+ |
| **GPU** | NVIDIA GPU strongly recommended (CUDA-capable PyTorch) |
| **Build tools** | CMake + C++ compiler — Kimodo builds **MotionCorrection** during install ([guide](docs/MOTION_CORRECTION.md)) |
| **Git** | Required for `pip` install (Kimodo is not on PyPI) |

Full setup (Blender addon, troubleshooting): **[Installation](docs/INSTALL.md)**.

## Quick start

```bash
pip install "motionmcp-kimodo @ git+https://github.com/animatica-ai/motionmcp-kimodo.git"
motionmcp-kimodo --port 8000
```

Install [Proscenium](https://github.com/animatica-ai/proscenium-blender/releases/latest), set **Server** to `http://localhost:8000`, and connect from the N-panel.

Install can take **30+ minutes** on first run (PyTorch, Kimodo, MotionCorrection compile). Use `pip install -v ...` to see progress.

## Documentation

| Guide | Description |
|---|---|
| [Installation](docs/INSTALL.md) | End-to-end setup with Proscenium |
| [MotionCorrection](docs/MOTION_CORRECTION.md) | C++ build step (common install blocker) |
| [Usage](docs/USAGE.md) | CLI, ports, models, environment variables |
| [Development](docs/DEVELOPMENT.md) | Architecture and programmatic use |
| [Docs index](docs/README.md) | All guides and external links |

## Related projects

| Project | Role |
|---|---|
| [MMCP](https://animatica.ai/mmcp) | Motion Model Communication Protocol |
| [motionmcp](https://github.com/animatica-ai/motionmcp) | Python SDK (`Backbone`, server helpers) |
| [Kimodo](https://github.com/animatica-ai/kimodo) | Motion diffusion model (dependency) |
| [Proscenium](https://github.com/animatica-ai/proscenium-blender) | Official Blender client |
| [Implementations](https://animatica.ai/mmcp/docs/get-started/implementations) | All official servers & clients |

## Contributing

Contributions are welcome.

1. Check [open issues](https://github.com/animatica-ai/motionmcp-kimodo/issues) or open one to discuss larger changes.
2. Clone, install dev deps, and run tests — see [Development](docs/DEVELOPMENT.md).
3. Open a pull request with a clear description and test plan.

Bug reports for install failures: include OS, Python version, and the full `pip install -v` log (especially the MotionCorrection build).

## Community

**[Animatica AI Discord](https://discord.com/invite/A8CrURBewz)** — questions, install help, Proscenium, and MMCP/Kimodo discussion.

## License

[Apache License 2.0](LICENSE). Kimodo model weights and third-party deps have separate licenses — see [Kimodo](https://github.com/animatica-ai/kimodo) and Hugging Face model cards.
