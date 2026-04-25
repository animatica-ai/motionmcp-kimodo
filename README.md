# motionmcp-kimodo

One of the **officially supported MMCP servers**. Wraps the
[Kimodo][kimodo] motion model in the [MMCP protocol][mmcp] — a drop-in
[`motionmcp.Backbone`][backbone] you can run on your own hardware.

Install from source:

```bash
git clone https://github.com/animatica-ai/motionmcp-kimodo
cd motionmcp-kimodo
pip install -e .
```

Run:

```bash
motionmcp-kimodo --port 8000
```

That's a working MMCP server for Kimodo on `:8000`. Point the
[Proscenium Blender plugin][proscenium] (the officially supported
client) at `http://localhost:8000` and you're animating.

[kimodo]: https://github.com/kimodo/kimodo
[proscenium]: https://github.com/animatica-ai/proscenium-blender
[backbone]: https://animatica.ai/mmcp/docs/sdk/backbone
[mmcp]: https://animatica.ai/mmcp
[impls]: https://animatica.ai/mmcp/docs/get-started/implementations

## What this package is

A thin glue layer:

- **Backbone**: [`KimodoBackbone`](src/motionmcp_kimodo/backbone.py)
  wraps the Kimodo model in `motionmcp.Backbone`. Loads weights, pre-builds
  a `ModelSpec`, runs `model(...)`, encodes the result as a `MotionResult`.
- **Translation**: [`translate.py`](src/motionmcp_kimodo/translate.py)
  converts an MMCP `GenerateRequest` into the kwargs the Kimodo model
  expects (prompts, segment durations, constraint sets).
- **Skeleton helpers**: [`skeleton.py`](src/motionmcp_kimodo/skeleton.py)
  converts a Kimodo skeleton object to the MMCP wire shape and resolves
  the four foot-contact channels to skeleton-specific joint names.

The [`motionmcp`][backbone] SDK does the rest — wire-format validation,
glTF encoding, error envelope, generic per-request checks.

## Usage

```bash
# Defaults: port 8000, default kimodo model, cuda:0 if available else cpu.
motionmcp-kimodo

# Pick a model + bind explicitly:
motionmcp-kimodo --model soma30 --port 8000 --device cuda:0

# Or run as a module:
python -m motionmcp_kimodo --model soma30
```

Clients pull the model's canonical skeleton from `/capabilities` and
send it verbatim in their `/generate` request — the wire format is
documented in the [MMCP docs][mmcp].

Environment overrides:

| Var | Effect |
|---|---|
| `KIMODO_MODEL` | Default model id when `--model` isn't passed |
| `TEXT_ENCODER_MODE` | Passed to `kimodo.load_model` (default `"dummy"` for development) |

## Programmatic use

If you want to mount Kimodo alongside other backbones, into a custom
FastAPI app, or behind your own ASGI deployment:

```python
from motionmcp import build_app, serve
from motionmcp_kimodo import KimodoBackbone

# Single backbone:
serve(KimodoBackbone(model_id="soma30"))

# Mount alongside others:
app = build_app({
    "kimodo-soma": KimodoBackbone(model_id="soma30"),
    "kimodo-fast": KimodoBackbone(model_id="soma30-fast"),
})
# then run with uvicorn server:app --workers 4 etc.
```

See the [`motionmcp` SDK docs][backbone] for the full Backbone surface.

## Status

Alpha. Tracks `motionmcp` 0.1.x and Kimodo 1.x.

## License

Apache 2.0.
