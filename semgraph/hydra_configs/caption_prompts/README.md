# caption_prompts — Hydra config group for VLM captioning prompt bundles

Pick a prompt bundle at the Hydra level to drive `semgraph/stages/caption.py`.

## Overview

Each YAML in this directory is a 2-line pointer that the `semgraph.prompting`
factory resolves to a concrete `PromptBundle` class. Bundle **content** lives
alongside the Python classes at `semgraph/prompting/data/<bundle_id>.yaml` so
hobbyists / companies / researchers can diff the actual prompts as a single
artifact.

## Built-in bundles

| `bundle_id` | Intended use                                  | `suggested_top_k` |
| ----------- | --------------------------------------------- | ----------------- |
| `standard`  | Baseline. Works on any capable VLM.           | –                 |
| `rich`      | Verbose prompts. Best for 2B+ VLMs.           | 5                 |
| `compact`   | Short prompts for smaller / faster VLMs.      | 3                 |
| `custom`    | Load a user-supplied YAML (see below).        | bundle-defined    |
| `default`   | Alias for `standard` (for Hydra defaults).    | –                 |

## Usage

```bash
# At the command line:
python -m semgraph.stages.caption ... caption_prompts=rich
```

```yaml
# In a user config overlay:
defaults:
  - batch_vlm_mapping_api
  - override caption_prompts: compact
```

## Migration from the legacy override

```
OLD: python -m semgraph.stages.caption ... +caption@caption=prompts_rich
NEW: python -m semgraph.stages.caption ... caption_prompts=rich
```

If you still reference `cfg.caption.prompts` / `cfg.caption.top_k` in a custom
overlay, the caption stage logs a one-shot deprecation warning with this exact
migration string.

## Custom bundles

Three ways to point `caption_prompts=custom` at a user YAML, in order of
precedence:

```bash
# 1. CLI (canonical idiom for interactive use):
python -m semgraph.stages.caption ... \
    caption_prompts=custom \
    caption_prompts.prompts_path=/abs/path/to/my_prompts.yaml

# 2. Env var (canonical idiom for sweep scripts):
export CAPTION_PROMPTS_FILE=/abs/path/to/my_prompts.yaml
python -m semgraph.stages.caption ... caption_prompts=custom

# 3. pkg:// URI (for installed third-party prompt packs):
python -m semgraph.stages.caption ... \
    caption_prompts=custom \
    caption_prompts.prompts_path=pkg://acme_prompts/kitchen.yaml
```

Filesystem paths must be inside the repo root, **or** the `trust_path`
flag / `CAPTION_PROMPTS_TRUST` env var must evaluate truthy per the case-
folded whitelist `{"1","true","yes","on"}`. `pkg://` paths skip that gate
because `importlib.resources` paths are structurally confined to installed
packages.

Guardrails on every custom bundle:

1. `yaml.safe_load` (no tag execution).
2. Path confinement or explicit trust flag (filesystem only).
3. Whitelisted top-level keys.
4. Placeholder audit — `{captions}` required in `consolidation`; no bare
   placeholders elsewhere.
5. Size caps: 8 KiB per template, 64 KiB per file.
6. SHA-256 hashed and logged at INFO level for reproducibility.

Validate a YAML before running the full pipeline:

```bash
python -m semgraph.prompting validate /abs/path/to/my_prompts.yaml
```

## Third-party prompt bundles

Register your own bundle class via setuptools entry points in your package's
`pyproject.toml`:

```toml
[project.entry-points."semgraph.prompt_bundles"]
acme_kitchen = "acme_prompts.bundles:KitchenPromptBundle"
```

Your class should subclass `semgraph.prompting.base.PromptBundle` and expose
a `from_cfg(cls, cfg) -> PromptBundle` classmethod.  Once installed
(`pip install acme-prompts`), `caption_prompts=acme_kitchen` just works.

Internal bundle IDs (`standard`, `rich`, `compact`, `default`, `custom`) are
reserved: a third-party registration on any of them gets a WARNING at load
time and the cache slot is overridden with the first-party class.
