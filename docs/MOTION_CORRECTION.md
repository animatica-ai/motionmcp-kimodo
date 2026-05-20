# MotionCorrection (C++ compile step)

Installing **motionmcp-kimodo** pulls in [Kimodo](https://github.com/animatica-ai/kimodo) as a dependency. Kimodo bundles **MotionCorrection** — a C++ library with Python bindings used for foot-skate cleanup and constraint post-processing.

**You do not install MotionCorrection separately.** It is compiled **on your machine** automatically when `pip` installs Kimodo. This step is often the slowest and most fragile part of setup.

[← Installation guide](INSTALL.md) · [Documentation index](README.md)

## What to expect

- During `pip install`, you will see CMake configure and a C++ compiler build `_motion_correction` (often several minutes).
- **Linux** is the best-tested platform. **macOS** (especially Apple Silicon) and **Windows** need extra build tools — see below.
- A failed compile stops the whole Kimodo install; fixing the toolchain and re-running `pip install` is the usual recovery.

## Prerequisites (install before `pip install`)

### All platforms

| Tool | Why |
|---|---|
| **[CMake](https://cmake.org/download/)** 3.15+ | Required. Kimodo’s installer calls `cmake` explicitly; if it is missing you get `CMake must be installed to build this package`. |
| **C++17 compiler** | Builds the extension (GCC, Clang, or MSVC). |
| **Python dev headers** | Usually included with Python on macOS/Windows; on Debian/Ubuntu install `python3-dev`. |

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev git
```

Fedora/RHEL equivalents: `gcc-c++`, `cmake`, `python3-devel`.

### macOS

1. Install **Xcode Command Line Tools** (provides Clang):

   ```bash
   xcode-select --install
   ```

2. Install **CMake** and, on **Apple Silicon (M1/M2/M3)**, **SIMDe** headers (required for ARM builds):

   ```bash
   brew install cmake simde
   ```

   If CMake cannot find SIMDe, point it at Homebrew’s include tree before installing:

   ```bash
   export CMAKE_PREFIX_PATH="$(brew --prefix)"
   pip install "motionmcp-kimodo @ git+https://github.com/animatica-ai/motionmcp-kimodo.git"
   ```

### Windows

Use **one** of these toolchains:

**Option A — Visual Studio (recommended)**

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or full Visual Studio with the **“Desktop development with C++”** workload.
2. Install [CMake](https://cmake.org/download/) and add it to PATH.
3. Open **“x64 Native Tools Command Prompt for VS”** (or a Developer PowerShell) and run `pip install` from there so MSVC is on PATH.

**Option B — MinGW**

Kimodo’s installer can use MinGW if `g++` is on PATH. You may also set:

```bat
set CMAKE_GENERATOR=MinGW Makefiles
```

MinGW runtime DLLs are copied next to the built module; if import fails later, ensure the same `g++` bin directory is on PATH.

## Install motionmcp-kimodo (after tools are ready)

From a terminal where `cmake --version` works:

```bash
pip install "motionmcp-kimodo @ git+https://github.com/animatica-ai/motionmcp-kimodo.git"
```

**Tip:** To see compile progress instead of a long silent wait:

```bash
pip install -v "motionmcp-kimodo @ git+https://github.com/animatica-ai/motionmcp-kimodo.git"
```

**Retry after a failed build** (clone gives clearer logs):

```bash
git clone https://github.com/animatica-ai/motionmcp-kimodo
cd motionmcp-kimodo
pip install -e . -v
```

## Verify MotionCorrection built correctly

```bash
python -c "import motion_correction; print('MotionCorrection OK')"
```

If that imports without error, the C++ extension is installed.

## Common errors

### `CMake must be installed to build this package`

Install CMake and ensure `cmake` is on your PATH (open a **new** terminal after installing).

### `SIMDe headers not found` (Apple Silicon)

```bash
brew install simde
export CMAKE_PREFIX_PATH="$(brew --prefix)"
```

Then re-run `pip install`.

### Compiler / linker errors mentioning `-msse4.1` or `-mavx`

You are on an x86_64 machine without AVX support, or using an unusual toolchain. Kimodo is primarily tested on recent Intel/AMD CPUs with AVX. Try a standard Linux box or a cloud GPU instance with a current Ubuntu image.

### `error: Microsoft Visual C++ 14.0 or greater is required`

Install the Visual Studio C++ build tools (see Windows above) and run `pip install` from a Developer Command Prompt.

### Build succeeds but `ImportError` for `_motion_correction` (Windows + MinGW)

Ensure MinGW `bin` (where `libstdc++-6.dll` lives) is on PATH, or reinstall using Visual Studio instead of MinGW.

### Install is slow — is it stuck?

MotionCorrection plus PyTorch/Kimodo downloads can take **30+ minutes** on first install. Use `pip install -v ...` to confirm the build is still running.

## Do I need MotionCorrection to run the MMCP server?

- **Install time:** Yes — Kimodo’s `pip` install always attempts to build MotionCorrection unless explicitly skipped (Docker/advanced only).
- **Runtime (motionmcp-kimodo):** Basic generation does not require post-processing. The server defaults to Kimodo’s dummy text encoder and typical MMCP requests do not enable `post_processing`. You still need a **successful compile** to complete `pip install`.

If you need Kimodo’s post-processing APIs (`post_processing=True`), MotionCorrection must import successfully (see verify step above).

## Still stuck?

1. **[Animatica AI Discord](https://discord.com/invite/A8CrURBewz)** — good place to ask about MotionCorrection / Kimodo build failures (paste your `pip install -v` log).
2. [Kimodo installation docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html) — official environment notes and Docker path.
3. [Kimodo virtual-env install](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation_virtual_env.html) — editable install from a cloned repo.
4. Open an issue on [motionmcp-kimodo](https://github.com/animatica-ai/motionmcp-kimodo/issues) or [animatica-ai/kimodo](https://github.com/animatica-ai/kimodo/issues) with the **full** `pip install -v` log from the MotionCorrection build failure.
