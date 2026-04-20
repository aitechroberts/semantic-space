# Phase B Runbook — Tuesday Sweep

End-to-end operator guide for the encoder × VLM sweep on Space3D-Bench.
Every stage is resumable: `.done` marker files guard each cell, JSON
artifacts are written via `atomic_write_json`, and `outputs/phase_b_logs/
sweep_progress.jsonl` is the single-line-per-cell progress ledger.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Layout & Conventions](#layout--conventions)
3. [Environment Variables](#environment-variables)
4. [Stages](#stages)
   - [S0 — Phase A preflight (already done)](#s0--phase-a-preflight)
   - [S1 — Download Space3D-Bench](#s1--download-space3d-bench)
   - [S2 — BigG Variant Synthesis](#s2--bigg-variant-synthesis)
   - [S3 — Encoder Sweep](#s3--encoder-sweep)
   - [S4 — VLM Captioning + Scene Graph](#s4--vlm-captioning--scene-graph)
   - [S5 — Attribute Vocabulary](#s5--attribute-vocabulary)
   - [S6 — Blind-LLM Question Vetting](#s6--blind-llm-question-vetting)
   - [S7 — VQA Evaluation (2-pass)](#s7--vqa-evaluation-2-pass)
5. [Top-Level Orchestration](#top-level-orchestration)
6. [Hardening Cheatsheet](#hardening-cheatsheet)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Dry-run the whole pipeline; prints what each stage would do.
DRY_RUN=1 bash generate_groundtruth/run_all.sh

# First wet smoke test: bigG synthesis with verification
VERIFY=1 bash generate_groundtruth/run_bigg_synthesis.sh

# Full sweep end-to-end (stops at the first failure).
bash generate_groundtruth/run_all.sh

# Resume after a crash (all stages honor .done markers).
bash generate_groundtruth/run_all.sh

# Verify outputs without running anything.
python generate_groundtruth/verify_gt_phase_a.py --phase-b
```

---

## Layout & Conventions

**Output tree** (rooted at `$OUTPUT_ROOT`, default `~/cmu-grad/neuro-experiments/ReplicaGroundTruth`):

```
<scene>/stages/
  oracle/                        # Phase A (input to Phase B)
    oracle_scene.npz
  crops/                         # Phase A crops
  variants/
    embed_<safe_encoder>/
      variant.npz
      .done
  captions/
    <safe_vlm>/
      captions.json
      .done
  scene_graphs/
    <safe_vlm>/                  # VLM-only scene graph (no encoder)
      scene_graph.json
      .done
  eval/
    <safe_enc>__sg_<safe_vlm>/
      clip_retrieval/{predictions,summary,per_question}.json + .done
    vlm_<safe_vlm>/
      full_sg/...
      task_subgraph_mst_vlm/...
      task_subgraph_flat_no_planes/...
```

**Logs** live under `outputs/phase_b_logs/<phase>/<scene>_<ident>.log`;
each cell appends one JSONL line to `outputs/phase_b_logs/sweep_progress.jsonl`.

**Space3D-Bench** (v0.0.2 release) extracts scene directories directly
into `$SPACE3D_ROOT/<scene_with_underscores>/` (i.e. `room_0/`,
`office_2/`, …). The GT file is called `ground_truth.json`
(**not** `answers.json` as older drafts suggested). `SCENE_MAP` in
`generate_groundtruth/_space3d_layout.py` is the single place that maps
our canonical scene ids (`room0`) to the on-disk form (`room_0`).

**Safe name mangling:** `vllm_safe_name` replaces `/` → `_` and `:` → `-`
so `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` becomes
`laion_CLIP-ViT-bigG-14-laion2B-39B-b160k`.

---

## Environment Variables

Shared across all sub-scripts (sensible defaults everywhere):

| Var | Default | Effect |
|---|---|---|
| `OUTPUT_ROOT` | `~/cmu-grad/neuro-experiments/ReplicaGroundTruth` | Phase A/B root |
| `DATASET_ROOT` | `~/cmu-grad/neuro-data/ReplicaNiceSLAM` | Input trajectories |
| `SPACE3D_ROOT` | `../Space3D-Bench` next to repo | Benchmark root |
| `SCENES` | `room0 room1 office2 office3` (shell-space-list) or `room0,room1,office2,office3` (csv for py) | Scene filter |
| `VLLM_PORT` | `8000` | API port |
| `FORCE` | `0` | Ignore `.done` markers |
| `DRY_RUN` | `0` | Print-only; no side effects |
| `PURGE_HF_CACHE` | `0` | `rm -rf` HF cache after each VLM (path-guarded) |
| `STOP_ON_FAIL` | `1` (run_all only) | Abort pipeline on first stage failure |

Script-specific variables are listed inline below.

---

## Stages

### S0 — Phase A Preflight

Not part of Phase B, but gate-check first:

```bash
python generate_groundtruth/verify_gt_phase_a.py
```

Expect **zero warnings** across all four scenes before touching Phase B.

---

### S1 — Download Space3D-Bench

**Preflight**

- Internet access; `git` + `unzip` + `wget` present.

**Run**

```bash
bash generate_groundtruth/download_space3d_bench.sh
# or: SPACE3D_ROOT=/custom/path bash generate_groundtruth/download_space3d_bench.sh
```

Idempotent: skips the clone if the repo exists and skips the zip if
any scene already has a `ground_truth.json`. Verifies that all 4
scenes have `questions.json` + `ground_truth.json` and prints the
question counts.

`DRY_RUN=1` short-circuits before any download — useful for
end-to-end plan printing.

**Saved artifacts**

- `$SPACE3D_ROOT/.git/…` (cloned repo)
- `$SPACE3D_ROOT/<room_0|room_1|office_2|office_3>/{questions,ground_truth}.json`
- `$SPACE3D_ROOT/<scene>/img/`  and  `$SPACE3D_ROOT/<scene>/misc/`

---

### S2 — BigG Variant Synthesis

**What it does.** BigG per-view features already live in
`oracle.pv_clip_ft_list` (Phase A). This stage reshapes them into a
Phase-B-style `VariantRecord` (weighted-average features, best-view
features, per-view entropy). It does **not** re-encode images. The
bigG *text tower* is loaded exactly once to encode the 50 Replica
labels for entropy, amortizing that load across all 4 scenes.

**Preflight**

- Phase A complete (`variants/` doesn't need to exist yet).
- GPU free (text tower runs briefly on `$DEVICE`, default `cuda`).
- Optional `VERIFY=1` requires `DATASET_ROOT` (so a reference
  `embed.mode=re_embed` can run for comparison).

**Run**

```bash
# standard
bash generate_groundtruth/run_bigg_synthesis.sh

# with allclose verification against a real re_embed on the first scene
VERIFY=1 bash generate_groundtruth/run_bigg_synthesis.sh

# force recompute
FORCE=1 bash generate_groundtruth/run_bigg_synthesis.sh

# dry run
DRY_RUN=1 bash generate_groundtruth/run_bigg_synthesis.sh
```

Under the hood this calls
`python generate_groundtruth/synthesize_bigg_variant.py --scenes room0,room1,office2,office3`.

**Saved artifacts (per scene)**

- `stages/variants/embed_laion_CLIP-ViT-bigG-14-laion2B-39B-b160k/variant.npz`
- `stages/variants/embed_laion_.../.done`
- Log: `outputs/phase_b_logs/phase2a_bigg/all_scenes.log`
- Ledger line: `{"phase":"phase2a_bigg",…}`

---

### S3 — Encoder Sweep

**What it does.** Per-crop feature extraction for the remaining
encoders. Uses `embed.mode=re_embed` which reads the same crops Phase A
produced and emits one `VariantRecord` per (encoder, scene). No vLLM.
No scene graph effect — the graph stays encoder-independent.

**Configurable encoders** (edit the `ENCODERS` bash array at top of
`run_encoder_sweep.sh`). Current defaults:

- `hf_clip | openai/clip-vit-large-patch14`
- `hf_siglip | google/siglip2-so400m-patch14-384`
- `open_clip | ViT-H-14:laion2b_s32b_b79k`
- `hf_clip | wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`

BigG is **not** in this list; it's S2.

**Preflight**

- `stages/oracle/oracle_scene.npz` + `stages/crops/*.jpg` exist.
- GPU memory budget ~24 GiB (enough for any single encoder).
- `EMBED_ENCODER_TYPE` + `EMBED_ENCODER` are threaded per-cell; no Hydra
  file edits needed.

**Run**

```bash
bash generate_groundtruth/run_encoder_sweep.sh

# purge HF cache after each encoder (disk-tight runs)
PURGE_HF_CACHE=1 bash generate_groundtruth/run_encoder_sweep.sh

# re-run a single scene
SCENES="room0" FORCE=1 bash generate_groundtruth/run_encoder_sweep.sh
```

**Saved artifacts (per (encoder, scene) cell)**

- `stages/variants/embed_<safe_encoder>/variant.npz`
- `stages/variants/embed_<safe_encoder>/.done`
- Log: `outputs/phase_b_logs/phase2b_encoder_sweep/<scene>_<safe>.log`
- Ledger: `{"phase":"phase2b_encoder_sweep","ident":"<safe>__<scene>",…}`

At the end, any `FAILED_CELLS` are listed and the script exits non-zero.

---

### S4 — VLM Captioning + Scene Graph

**What it does.** For each VLM:

1. Bring vLLM server up (once per VLM).
2. For each scene:
   a. Run `semgraph.stages.caption` with `top_k=5` and the
      `prompts_rich` bundle (2-3 sentence object prompts).
   b. Immediately run `assemble_scene_graph.py`, producing a
      **VLM-only** scene graph (no encoder dependency).
3. Tear vLLM down, optionally purge HF cache, next VLM.

The scene graph file contains: `scene_type` (VLM-inferred), `objects`
(geometry from oracle + caption / tag / color / material from VLM),
`planes` (HPSG), and `edges` (MST + VLM-labeled spatial relations).

**Configurable VLMs** (edit `VLMS` in `run_vlm_captioning_sweep.sh`):

- `Qwen/Qwen3-VL-2B-Instruct` (gpu=0.75, mml=3072)
- `OpenGVLab/InternVL3-2B` (gpu=0.75, mml=4096)
- `google/gemma-3-4b-it` (gpu=0.55, mml=2048) — note the tighter VRAM.
- `AIDC-AI/Ovis2.5-2B` (gpu=0.70, mml=3072) — dry-run the first cell if
  Ovis has issues on your vLLM version.

**Preflight**

- `stages/oracle/` + `stages/crops/` exist.
- No other process holding the GPU (`vllm_check_orphan` runs at start).
- `sentence-transformers` installable (only needed if you plan to skip
  ahead to S7 on the same box; S4 itself doesn't need it).

**Run**

```bash
bash generate_groundtruth/run_vlm_captioning_sweep.sh

# swap back to the old short prompts
PROMPT_BUNDLE=prompts_standard \
  bash generate_groundtruth/run_vlm_captioning_sweep.sh

# purge cache between VLMs
PURGE_HF_CACHE=1 bash generate_groundtruth/run_vlm_captioning_sweep.sh

# dry run just the plan
DRY_RUN=1 bash generate_groundtruth/run_vlm_captioning_sweep.sh
```

**Saved artifacts (per (VLM, scene) cell)**

- `stages/captions/<safe_vlm>/captions.json` + `.done`
- `stages/scene_graphs/<safe_vlm>/scene_graph.json` + `.done`
- Logs: `outputs/phase_b_logs/phase3_captioning_sweep/<scene>_<safe>_{caption,sg}.log`
- Ledger: two lines per cell (`phase3_caption`, `phase3_scene_graph`).

---

### S5 — Attribute Vocabulary

**What it does.** Parses every `ground_truth.json` into an
informational answer vocabulary. v0.0.2 has no categorical `type`
field, so `_space3d_layout.infer_type` classifies each entry from the
answer text itself:

| Format                                       | Inferred type     | Scoring support |
|----------------------------------------------|-------------------|-----------------|
| `Yes` / `No`                                 | `binary`          | ✓ exact         |
| `Objects: ['book', 'candle', ...]`           | `object_list`     | ✓ multiset      |
| `dict{image_path, example_answer}` or free   | `qualitative`     | ✓ weak substring |
| `Number of objects: N`                       | `count`           | ✗ format_unsupported |
| `3D position: [[x, y, z], ...]`              | `position`        | ✗ format_unsupported |
| `From object=... navigable distance ...`     | `distance`        | ✗ format_unsupported |

The vocab script is informational — the deterministic scorer in
`vqa_eval.py` does not depend on the file, but having it on disk lets
humans and downstream tooling inspect exactly which tokens could match.

**Preflight**

- S1 complete (`$SPACE3D_ROOT/<scene>/ground_truth.json` exists).

**Run**

```bash
# Part of run_all.sh (stage s5_vocab). Standalone:
python generate_groundtruth/build_attribute_vocab.py \
    --space3d_root "$SPACE3D_ROOT" \
    --scenes room0,room1,office2,office3
```

**Saved artifacts (per scene)**

- `$SPACE3D_ROOT/<scene>/attribute_vocab.json`
  - `.binary`        — `["yes", "no"]`
  - `.object_list`   — object tokens from `Objects: [...]` answers, freq-desc
  - `.qualitative`   — content tokens from qualitative answers, freq-desc
  - `.unsupported`   — counts per unsupported type
  - `.per_question`  — `qid → inferred_type`
  - `.n_answers`     — total entries

---

### S6 — Blind-LLM Question Vetting

**What it does.** Filtering mechanism (not a VQA method). Stands up a
single text-only LLM via vLLM, asks every `(scene, question)` **with no
image or scene context**, 5 rolls at `T=0.7`. Scoring uses the same
`score_prediction` helper as S7, so agreement is judged per the
question's inferred type:

- **binary** yes/no — **all 5** rolls must match (unanimity),
- **object_list / qualitative** — **≥ 3 of 5** rolls must match (majority),
- **format_unsupported** (`count`/`position`/`distance`) — always
  `scored=False` and `language_only_correct=False`; rolls are still
  recorded for reproducibility.

Questions are **not removed**; VQA accuracy is later stratified into
`all` and `novel` (= `!language_only_correct`).

**Preflight**

- S1 complete.
- GPU free.
- `openai` Python package (already in uv.lock).

**Run**

```bash
bash generate_groundtruth/run_question_vetting.sh

# with a different LLM
VET_MODEL=Qwen/Qwen2.5-7B-Instruct VET_MML=4096 VET_GPU_MEM=0.70 \
  bash generate_groundtruth/run_question_vetting.sh

# force re-vetting
FORCE=1 bash generate_groundtruth/run_question_vetting.sh
```

**Saved artifacts (per scene)**

- `$SPACE3D_ROOT/<scene>/vetted_questions.json`
  - `.per_question[qid].rolls` (5 strings)
  - `.per_question[qid].agreements` (0–5)
  - `.per_question[qid].type` (`binary|object_list|qualitative|count|position|distance`)
  - `.per_question[qid].scored` (bool)
  - `.per_question[qid].threshold` (5 for binary, 3 otherwise)
  - `.per_question[qid].language_only_correct` (bool)
  - `.per_question[qid].has_negation` (bool — reporting stratum)
  - `.summary.by_type[<qtype>]` with `n / scored / language_only`
- `$SPACE3D_ROOT/<scene>/vetted_questions.done` marker
- Log: `outputs/phase_b_logs/phase4_vetting/vet_<safe_model>.log`

---

### S7 — VQA Evaluation (2-pass)

**Pass 1 — Encoder Pass (vLLM-free).** Runs `clip_retrieval` for every
(encoder, scene). The encoder's text tower encodes the **full question
text**; the top-1 object by cosine against `clip_ft_weighted_avg`
(override with `FEATURE_KEY`) is returned. The prediction is just that
object's `object_tag` — one string, no attribute heuristics, no
counting heuristics, no noun extraction.

**Pass 2 — VLM Pass.** For each VLM, bring vLLM up, and run three
methods against the VLM-only scene graph:

- `full_sg` — feed the whole compact scene graph to the VLM.
- `task_subgraph_mst_vlm` — Sparse3DPR-inspired:
  seed `SEED_K=5` nearest objects by MiniLM caption embedding (τ=0.07),
  expand 2-hop along **MST ∪ VLM edges**, feed resulting subgraph to
  the VLM with the `sparse3dpr_{yesno,counting,descriptive}.txt`
  prompts.
- `task_subgraph_flat_no_planes` — same as above, but plane nodes and
  their incident edges are dropped (no plane → object promotion).

Deterministic scorer (`_space3d_layout.score_prediction`) handles only
the three supported answer formats; unsupported formats
(`count`/`position`/`distance`) are tagged `format_unsupported` and
**excluded from the accuracy denominator** (they are preserved in
`per_question.json`).

Scoring rules per supported type:

- `binary`:       first-token yes/no match (with yes↔true, no↔false aliases).
- `object_list`:  multiset containment against the GT `Objects: [...]`
                  list. Prediction is tokenized and compared token-set
                  style — order-irrelevant.
- `qualitative`:  weak content-token substring match. Accept if ≥ 1/3
                  of the GT's content tokens (len ≥ 4, non-stopword)
                  appear in the prediction. Reported as such in
                  `per_question[qid].reason`.

Accuracy strata in `summary.json`:

- `all`               — every **scored** question (denominator excludes
                         `format_unsupported`).
- `novel`             — scored & `language_only_correct == False`.
- `by_type[<qtype>]`  — one of `binary | object_list | qualitative`.
- `by_negation[with|without]`.
- `format_unsupported.total` and `.by_type` — count only, reported but
  not in the denominator.

**Preflight**

- S2, S3, S4 done (variants + scene graphs).
- S5, S6 done (attribute vocab + vetted questions for `novel` stratum).
- `sentence-transformers` installed (used for Pass 2 only). The
  `pyproject.toml.backup` now lists it; apply with `uv sync` when you
  restore the real `pyproject.toml`.

**Run**

```bash
# both passes, all encoders × VLMs
bash generate_groundtruth/run_vqa_eval_sweep.sh

# only encoder pass (no vLLM needed)
DO_VLM_PASS=0 bash generate_groundtruth/run_vqa_eval_sweep.sh

# only VLM pass
DO_ENCODER_PASS=0 bash generate_groundtruth/run_vqa_eval_sweep.sh

# different scene graph source for encoder pass
SCENE_GRAPH_VLM=OpenGVLab/InternVL3-2B \
  bash generate_groundtruth/run_vqa_eval_sweep.sh

# knobs
SEED_K=7 TAU=0.1 FEATURE_KEY=clip_ft_best \
  bash generate_groundtruth/run_vqa_eval_sweep.sh
```

**Saved artifacts**

Encoder pass, per (encoder, scene, scene_graph_vlm):

- `stages/eval/<safe_enc>__sg_<safe_vlm>/clip_retrieval/{predictions,summary,per_question}.json + .done`

VLM pass, per (VLM, scene, method):

- `stages/eval/vlm_<safe_vlm>/full_sg/…`
- `stages/eval/vlm_<safe_vlm>/task_subgraph_mst_vlm/…`
- `stages/eval/vlm_<safe_vlm>/task_subgraph_flat_no_planes/…`

Logs: `outputs/phase_b_logs/phase5_vqa_eval/*.log`

Ledger: `{"phase":"phase5_vqa_eval","ident":"encoder_pass"|"vlm_pass__<safe>"}`.

---

## Top-Level Orchestration

```bash
# Default: run all stages in order, abort on first failure.
bash generate_groundtruth/run_all.sh

# Restrict stages.
STAGES="s2_bigg s3_encoders s4_captions" bash generate_groundtruth/run_all.sh

# Soldier on past failures.
STOP_ON_FAIL=0 bash generate_groundtruth/run_all.sh

# End-to-end dry run.
DRY_RUN=1 bash generate_groundtruth/run_all.sh
```

Stages: `s1_space3d, s2_bigg, s3_encoders, s4_captions, s5_vocab, s6_vetting, s7_vqa`.

Every invocation writes to `outputs/phase_b_logs/run_all/<stage>.log`
and appends to the shared ledger, so you can track wall-clock cost of
each stage with:

```bash
# Quick cost breakdown.
tail -n 1000 outputs/phase_b_logs/sweep_progress.jsonl \
  | jq -r '[.phase, .ident, .status, .elapsed_s] | @tsv' | sort
```

---

## Hardening Cheatsheet

- **`.done` markers** — every sub-script checks for `<out_dir>/.done`
  before running and touches it on success. Delete the marker to force
  a re-run.
- **`atomic_write_json`** — all new JSON artifacts use write-tmp +
  `os.replace`. A crash mid-write leaves the previous artifact intact.
- **`FAILED_CELLS`** — every sweep script accumulates failures into a
  bash array; at the end it prints them and exits non-zero. One bad
  cell does not abort the rest of the sweep.
- **`vllm_check_orphan`** — runs at the top of every sweep that starts
  vLLM; `pgrep -f "vllm serve"`, then `pkill -TERM`, then `pkill -KILL`
  if needed.
- **`vllm_down`** — `SIGTERM` → 15 s poll → `SIGKILL` → `pkill -P`
  child reap → `nvidia-smi` VRAM polling until below
  `VRAM_DOWN_MIB=2048` MiB or `VRAM_POLL_S=60` s timeout.
- **`vllm_purge_cache`** — path-guarded `rm -rf` only under
  `${HF_HOME}/hub/models--*`. Refuses anything else.
- **`vllm_safe_name`** — `/` → `_`, `:` → `-`. Used everywhere safe
  filenames are needed.
- **`sanity_sample`** — dumps N random entries from any JSON artifact
  for eyeball inspection (`sanity_sample path/to/captions.json 5`).
- **`DRY_RUN=1`** — every script honors it; nothing is written, but
  every planned command is printed.

---

## Troubleshooting

### vLLM won't come healthy

Check the log under `outputs/phase_b_logs/phase<N>_*/<scene>_<safe>.log`.
The last 50 lines of the server log are copied on failure. Common
causes: out-of-disk HF cache, corrupted model snapshot, mismatched
`max-model-len` for your VRAM.

Run `pgrep -af 'vllm serve'`. If there's a zombie, `pkill -KILL -f
'vllm serve'` and re-run; `vllm_check_orphan` also does this at
start.

### Encoder re_embed fails with "skip_matching"

Phase A should have been run with `skip_matching=True` (gt_instances
mode). If not, the oracle's `per_view_records` are undefined and
re-embed has no crops to work with. Re-run Phase A, do not patch
Phase B.

### BigG `--verify` reports a mismatch

The synthesized variant and a reference `embed.mode=re_embed` should
agree to 1e-4 per-view. If they don't:

1. Check that `oracle.pv_clip_ft_list` and `oracle.per_view_meta` have
   the same length per object (the synth script truncates to the
   shorter of the two).
2. Make sure `config/replica_50_labels.txt` hasn't changed between the
   Phase A run and the verification.

### VQA accuracy is 0% for `clip_retrieval` on an encoder

- Confirm `stages/variants/embed_<safe>/variant.npz` exists and its
  feature dim matches the encoder's text-tower dim (dimension mismatch
  is logged and short-circuits).
- The questions are encoded directly — no noun extraction — so shorter
  tokenizers (TinyCLIP 77-token cap) can truncate long questions; that
  is expected and reported.

### `sentence-transformers` ImportError

S7 VLM pass needs `sentence-transformers`. The dependency is listed in
`pyproject.toml.backup`; apply it with `uv add sentence-transformers`
or restore `pyproject.toml` and `uv sync`.

### Resuming after a partial run

Just re-run the same command. Every stage checks `.done` markers and
skips completed cells. Check which cells failed:

```bash
grep '"status":"fail"' outputs/phase_b_logs/sweep_progress.jsonl \
  | jq -r '[.phase, .ident, .log] | @tsv'
```

Delete the `.done` marker (or pass `FORCE=1`) to force a re-run of a
specific cell.

### Verify Phase B outputs without running anything

```bash
python generate_groundtruth/verify_gt_phase_a.py --phase-b
# custom encoder/VLM set:
python generate_groundtruth/verify_gt_phase_a.py --phase-b \
  --encoders "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k,openai/clip-vit-large-patch14" \
  --vlms "Qwen/Qwen3-VL-2B-Instruct"
```
