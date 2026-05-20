# Development

How **motionmcp-kimodo** is structured and how to embed it in your own stack.

[← Documentation index](README.md)

## What this package is

A thin glue layer around [Kimodo](https://github.com/animatica-ai/kimodo) and the [`motionmcp` Backbone SDK](https://animatica.ai/mmcp/docs/sdk/backbone):

| Module | Role |
|---|---|
| [`backbone.py`](../src/motionmcp_kimodo/backbone.py) | `KimodoBackbone` — loads weights, builds `ModelSpec`, runs `model(...)`, returns `MotionResult` |
| [`translate.py`](../src/motionmcp_kimodo/translate.py) | Maps MMCP `GenerateRequest` → Kimodo kwargs (prompts, durations, constraints) |
| [`skeleton.py`](../src/motionmcp_kimodo/skeleton.py) | Kimodo skeleton → MMCP wire shape; foot-contact joint names |

The SDK handles wire validation, glTF encoding, error envelopes, and generic per-request checks.

## Programmatic use

Mount Kimodo alongside other backbones, in a custom FastAPI app, or behind your own ASGI deployment:

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
# then: uvicorn server:app --workers 4
```

See the [`motionmcp` SDK docs](https://animatica.ai/mmcp/docs/sdk/backbone) for the full Backbone surface.

## Install for development

```bash
git clone https://github.com/animatica-ai/motionmcp-kimodo
cd motionmcp-kimodo
pip install -e ".[dev]"
pytest
```

Kimodo is a git dependency (see `pyproject.toml`); it is not published to PyPI.

## Status

Alpha. Tracks `motionmcp` 0.1.x and Kimodo 1.x.
