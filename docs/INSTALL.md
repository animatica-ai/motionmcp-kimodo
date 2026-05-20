# Installation

End-to-end setup for **self-hosted Kimodo** with **[Proscenium for Blender](https://github.com/animatica-ai/proscenium-blender)**.

You need **two pieces**:

1. **This motion server** — runs in a terminal on your machine (GPU-heavy AI work).
2. **Proscenium** — Blender addon that sends requests and bakes animation onto your rig.

[← Documentation index](README.md)

## What you need first

| Requirement | Notes |
|---|---|
| **Computer with a recent NVIDIA GPU** | Strongly recommended. Generation on CPU alone is possible but very slow. |
| **Python 3.10 or newer** | [python.org/downloads](https://www.python.org/downloads/) — on Windows, check **“Add python.exe to PATH”** during install. |
| **Git** | [git-scm.com/downloads](https://git-scm.com/downloads) — Kimodo is installed from GitHub, not the usual Python package index. |
| **C++ build tools + CMake** | Kimodo compiles **MotionCorrection** on your machine during `pip install`. See **[MotionCorrection guide](MOTION_CORRECTION.md)** — install CMake and a compiler *before* step 1. |
| **Disk space & internet** | First run downloads Kimodo model weights from Hugging Face (several GB). |
| **Blender 4.x** | For the Proscenium addon (step 3 below). |

> **Biggest install hurdle:** MotionCorrection is a C++ extension bundled with Kimodo. It is not a separate download — `pip` builds it locally and the step often takes many minutes. Platform-specific prerequisites (Linux / macOS / Windows): **[MotionCorrection](MOTION_CORRECTION.md)**.

## 1. Install the motion server

Open a **terminal** (macOS: Terminal.app → search “Terminal”; Windows:
Start → “Command Prompt” or “PowerShell”).

### Option A — one command (easiest)

Install **CMake and a C++ toolchain first** ([MotionCorrection guide](MOTION_CORRECTION.md)), then paste this and press Enter. The first run can take **30+ minutes** while it downloads PyTorch, **compiles MotionCorrection**, and installs Kimodo:

```bash
pip install "motionmcp-kimodo @ git+https://github.com/animatica-ai/motionmcp-kimodo.git"
```

To watch the C++ build instead of a long silent wait, use `pip install -v ...` (same URL as above).

### Option B — clone the repo

Use this if you want to edit code or retry a failed install:

```bash
git clone https://github.com/animatica-ai/motionmcp-kimodo
cd motionmcp-kimodo
pip install -e .
```

Kimodo is pulled from [animatica-ai/kimodo](https://github.com/animatica-ai/kimodo) (not PyPI). If install fails while **building MotionCorrection** (CMake / compiler errors), see **[MotionCorrection](MOTION_CORRECTION.md)** first, then [Troubleshooting](#troubleshooting).

## 2. Start the server

In the same terminal:

```bash
motionmcp-kimodo --port 8000
```

Leave this window **open** while you work in Blender. When startup finishes you should see a line like `ready` and the process listening on port **8000**. The first start also downloads model weights — wait until that completes.

**You’re done with the server side** when Blender can reach `http://localhost:8000` (step 4).

CLI options and environment variables: **[Usage](USAGE.md)**.

## 3. Install Proscenium in Blender

1. Download the latest **`proscenium-blender-*.zip`** from [Proscenium releases](https://github.com/animatica-ai/proscenium-blender/releases/latest).
2. In Blender: **Edit → Preferences → Add-ons → Install…** → pick the zip.
3. Enable **Proscenium — AI Motion Generation** in the addon list.

[Video tutorials](https://www.youtube.com/watch?v=Wc349qOwjfM&list=PLAJ2UfUYhFQKZpFS8eh1eGUWJ0PAys1n1) · [Proscenium README](https://github.com/animatica-ai/proscenium-blender#readme)

## 4. Connect Blender to your server

1. In Blender: **Edit → Preferences → Add-ons → Proscenium → Preferences**
2. Set **Server** to: `http://localhost:8000`
3. In the 3D viewport **N** panel → **Proscenium** tab → **Connect**

If Connect fails, the server is not running or something is blocking port 8000 — see [Troubleshooting](#troubleshooting).

## 5. Generate motion

1. **Import model skeleton** (Proscenium button — or use a matching rig; see Proscenium docs)
2. Add prompts / constraints on the timeline
3. Click **Generate Motion**

## Troubleshooting

### `pip` or `python` not found

- Reinstall Python from [python.org](https://www.python.org/downloads/) and enable **Add to PATH** (Windows).
- On macOS/Linux, try `python3` and `pip3` instead of `python` / `pip`.

### Install fails while building MotionCorrection or Kimodo

This usually means the **C++ compile step** failed, not Python itself.

1. Follow **[MotionCorrection](MOTION_CORRECTION.md)** for your OS (CMake, compiler, Apple Silicon SIMDe, Windows MSVC).
2. Verify: `python -c "import motion_correction; print('OK')"`.
3. Make sure **Git** is on your PATH.
4. Read the [Kimodo installation guide](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html) for CUDA / PyTorch notes.
5. Retry with verbose logs: `pip install -v -e .` from a cloned `motionmcp-kimodo` folder.

### `motionmcp-kimodo` not found after install

- Close and reopen the terminal, then try again.
- Or run: `python -m motionmcp_kimodo --port 8000`

### Blender “Connect” fails

- Confirm the server terminal still shows the process running (no error traceback).
- Server URL must be exactly `http://localhost:8000` (no trailing path).
- Firewall: allow local connections on port 8000.

### Out of memory (GPU)

- Use a shorter clip or fewer constraints.
- Close other GPU apps (games, other ML tools).
- For Kimodo’s full local text encoder (not the server default), see [Usage](USAGE.md) and Kimodo’s `TEXT_ENCODER_DEVICE=cpu` notes in the [Kimodo docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html).

### Need help?

Ask on the **[Animatica AI Discord](https://discord.com/invite/A8CrURBewz)** (install issues, Blender, server setup). You can also [open an issue](https://github.com/animatica-ai/motionmcp-kimodo/issues) on GitHub.

### Don’t want to self-host?

Use **Animatica Cloud** instead — same MMCP protocol, no local server setup. See [official implementations](https://animatica.ai/mmcp/docs/get-started/implementations).
