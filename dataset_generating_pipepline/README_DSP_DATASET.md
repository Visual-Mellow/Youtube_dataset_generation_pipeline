# Dataset pipeline reference

Quick start and one-command runs live in the repository [README](../README.md). This file explains **how the cue pipeline actually works**: Layer-0 DSP anchors, Gemini agents/prompts, LOW vs HIGH, and the artifacts each stage writes.

Core idea: **timing is owned by local DSP; Gemini only names audible events.** Annotations describe what is audible in the source — not a Foley cue sheet invented from picture alone.

---

## Layer-0 v2 (default: `anchor_version: 2`)

DSP upgrade while keeping the Gemini-as-namer invariant. Toggle with `--anchor-version 1|2` (v1 path kept for A/B).

| Piece | What it does |
| --- | --- |
| `perception/timebase.py` | ffprobe A/V `start_time`, VFR flag, per-frame PTS (`frame_pts_s.npy`); never `frame_index/fps` |
| `perception/onsets_v2.py` | Multi-band SuperFlux @ 32 kHz + Hilbert refine + decay offsets |
| `perception/sed/` | Pluggable SED (PANNs DecisionLevel / windowed tagger); SED = class+extent, DSP = edges |
| `perception/video_cues.py` | Frame-diff + Farneback + **mean pixel-intensity changes** (`|Δ|≥5`) → `intensity_change` anchors for the LLM |
| `perception/avsync.py` | Onset↔motion lag estimate; warn-only unless `--apply-av-offset` |
| Fusion schema **2.1.0** | Additive `t_ms_f`, `t_end_ms_f`, `bands`, `sed`, `score`, `visual_support`, … |
| `annotate/snap.py` v2 | Class-aware Hungarian assignment; INSTANT/EVENT ends from DSP when reliable |

Eval / gold / ablations: [`eval/`](eval/), notes in [`docs/layer0_v2_ablation.md`](docs/layer0_v2_ablation.md).

```bash
# Synthetic onset gate
python -m tests.test_synthetic

# Baseline vs gold (empty gold → writes empty baseline JSON)
python -m eval.run_eval --features-dir ../Dataset/extracted/features --anchor-version 1

# Debug plot
python -m tools.plot_anchors --features-dir ../Dataset/extracted/features \
  --video-id g-6g2uEjF1s --clip-id clip_001 --level high
```

---

## End-to-end flow (one clip)

```text
raw clip MP4
    │
    ├─ validate A/V
    ├─ VideoMAE tokens          → video_tokens.pt  (features for downstream models)
    ├─ Layer 0 perception       → perception/anchors.json
    │     timebase / onsets_v2 / SED / video_cues / beats / Demucs(HIGH) / ASR
    │
    └─ Annotate (level=low|high) → annotations/<level>/
          1. Explanation agent   (Gemini + full A/V clip)
          2. Proposer agent(s)   (slot-fill cues onto anchors)
          3. Rule verifier + snap_v2 (DSP retimes; reject not-audible)
          4. Judge agent (HIGH)  (keep / drop / relabel — no new times)
          → cue_annotations.json, record.json, pipeline.log, pipeline.ui.log
```

Entry point:

```bash
python dataset_generating_pipepline/build_dataset.py \
  --video-id g-6g2uEjF1s --clip-id clip_001 --processing-level high \
  --anchor-version 2
```

Config: [`core/pipeline_config.json`](core/pipeline_config.json) + [`core/config.py`](core/config.py) + [`core/schemas.py`](core/schemas.py). CSV column `processing_level` overrides the CLI default.

### Package modules (after restructure)

| Folder | Role |
| --- | --- |
| `core/` | Path constants, dataclasses, `load_pipeline_config()` |
| `video_helpers/` | Download, scene/fixed segmentation, `extract_video` wrapper |
| `perception/` | Layer-0 DSP (onsets, SED, video cues, timebase, …) |
| `annotate/` | Gemini explain / propose / verify / judge |
| `utils/` | CSV loaders, UI logs, logger |
| `orchestrator.py` | Batch download → extract wiring used by `run_youtube_links.py` |
| `build_dataset.py` | Per-clip feature + annotation entry point |

---

## LOW vs HIGH

| Level | What runs | Extra API cost |
| --- | --- | --- |
| `low` (default) | Layer 0 (no Demucs) + 1 explanation + 1 whole-clip A/V proposer + rule verifier/snap | ≈ 2 Gemini flash calls |
| `high` | Layer 0 + Demucs stems + per-window A/V + audio-only proposers + pro judge | ~10–14 flash + 1 pro per ~30 s clip |

Flags (from config):

| Flag | LOW | HIGH |
| --- | --- | --- |
| `separation` (Demucs) | off | on |
| `per_window_proposer` | off (one whole-clip call) | on (~6 s windows, cut-snapped) |
| `audio_only_pass` | off | on (second pass per window) |
| `llm_judge` | off | on (`gemini-3.1-pro-preview`) |
| `onsets_v2` + refine | on (when `anchor_version=2`) | on |
| `sed` | `windowed_tagger` | `panns_cnn14_sed` on effects stem |
| `snap_v2` | on | on |

### Concrete HIGH example (`g-6g2uEjF1s/clip_001`)

From `annotations/high/pipeline.ui.log` / `llm_calls.jsonl`:

| Stage | Result |
| --- | --- |
| Perception | **114 anchors** (75 audio_onset, 8 cut, 22 motion, 3 speech_start, 6 music_boundary); Demucs on |
| Explanation | 1× flash → `video_type=ACTION` |
| Proposer | 6 windows × (A/V + audio-only) = **12** flash calls → **84** raw cues |
| Verifier + judge | rules + 1× pro → **51** kept, **33** rejected |
| Total | **14** LLM calls, ~336 s annotate time |

View the structured log:

```bash
python -m tools.view_ui_log \
  ../Dataset/extracted/features/g-6g2uEjF1s/clip_001/annotations/high/pipeline.ui.log
```

---

## Layer 0 — how anchors are made

Code: [`perception/anchors.py`](perception/anchors.py). Output: `features/<vid>/<clip>/perception/anchors.json` (shared across LOW/HIGH; cache keyed by clip SHA-256 + `separation_used`).

### 1. Decode audio

Clip → mono PCM at 16 kHz (`perception/audio_io.decode_mono`).

### 2. Optional stem separation (HIGH)

[`perception/separation.py`](perception/separation.py) runs Demucs `htdemucs` and maps stems:

| Pipeline stem | Source |
| --- | --- |
| `dialogue` | vocals |
| `music` | drums + bass + other |
| `effects` | residual (mixture minus speech/percussion proxy) |
| `mixture` | original |

Stem WAVs may be written under `perception/stems/`. If Demucs is missing, perception falls back to mixture-only onsets.

### 3. Audio onsets (the SFX clock)

[`perception/onsets.py`](perception/onsets.py) — mel spectral-flux peak picking:

1. STFT → 64-band mel spectrogram → log magnitude  
2. Per-frame **positive** first difference summed across bands = onset strength  
3. Peak-pick: local max in ±30 ms, above local mean (±200 ms) + `delta` (default **0.6**), and ≥ `min_gap_ms` (default **100 ms**) after the previous peak  
4. Hop = **10 ms**

**LOW:** onsets on the mixture only (`stem=mixture`).

**HIGH:** onsets on `effects`, `dialogue`, and `music` stems; keep mixture onsets that are not already covered within `dedupe_ms` (40 ms). Effects-stem onsets are the primary SFX timing source for later snap.

Each onset becomes:

```json
{
  "t_ms": 70,
  "kind": "audio_onset",
  "stem": "effects",
  "strength": 2.27,
  "on_music_beat": false,
  "tagger_topk": [["Music", 0.76], ...]
}
```

### 4. Beats

[`perception/beats.py`](perception/beats.py) estimates tempo + beat grid; each onset gets `on_music_beat` if within 50 ms of a beat. Beat times are stored on the anchors doc as `beats_ms` / `tempo_bpm`.

### 5. Audio tagger

[`perception/tagger.py`](perception/tagger.py) tags the clip (and tags near each onset) for later naming hints — not for timing.

### 6. Visual anchors (from video)

[`perception/scenes.py`](perception/scenes.py):

- **Cuts** — ffmpeg `scene_score` at `anchor_fps` (15): frames with score ≥ 0.35 → `kind=cut`
- **Motion peaks** — local maxima of non-cut scene scores above mean + 1.5σ → `kind=motion`

(PySceneDetect ContentDetector is used for **clip segmentation** at download time, not for these per-clip cut/motion anchors.)

### 7. Speech anchors

[`perception/speech.py`](perception/speech.py) — faster-whisper word timestamps → `speech_start` anchors with `transcript` / `end_ms`. These can seed `DIALOGUE` cues without Gemini inventing speech times.

### 8. Music boundaries

MFCC self-similarity novelty on the music stem (or mixture): coarse section changes → `music_boundary` (min 2 s apart). Used when snapping AMBIENCE/MUSIC beds.

### 9. Merge + enrich

All events are merged, deduped within `dedupe_ms` (prefer speech > audio_onset > music_boundary > cut > motion), enriched with `near_cut_ms` / `near_motion_peak_ms`, sorted, and given stable ids `a000`, `a001`, …

Agents never invent the clock: they receive a text dump of these anchors via `format_anchors_for_prompt()` (capped per window, prefer speech/onsets).

---

## Agents — who does what, and what prompt they get

All prompts live in [`annotate/prompts.py`](annotate/prompts.py). Orchestration: [`annotate/orchestrator.py`](annotate/orchestrator.py). Every Gemini call is logged to `llm_calls.jsonl` + raw JSON under `llm/`.

### 1. Explanation agent (LOW + HIGH)

| | |
| --- | --- |
| **Model** | `models.explain` → `gemini-3-flash-preview` |
| **Input** | Full unmuted clip MP4 + `EXPLANATION_PROMPT` |
| **Job** | Watch/listen; write beat sheet + sonic world. Does **not** emit cue timestamps. |
| **Output** | `gemini_explanation.json` → `video_content`, `scene_gravity`, `scene_interpretation`, `audio_asthetics`, `video_type` |

Framing: audio is ground truth; visuals explain *why* a sound exists; do not invent inaudible sounds.

`video_type` ∈ `ACTION`, `DIALOGUE`, `ESTABLISHING`, `CHASE`, `FIGHT`, `REVEAL`, `EMOTIONAL`, `HORROR_TENSION`, `TRAVEL`, `OTHER`.

### 2. Proposer agent(s) — slot-filling

| | |
| --- | --- |
| **Model** | `models.proposer` → `gemini-3-flash-preview` |
| **Prompt** | `PROPOSER_PROMPT` (A/V) or `AUDIO_ONLY_PROPOSER_PROMPT` (audio WAV only) |
| **Injected context** | window clock, compact explanation JSON as `visual_story`, Layer-0 anchor list |
| **Job** | For each `audio_onset` / `speech_start`: name it (`audio_class`) or `"none"`. May add missed events only if confident within ±200 ms. Emit continuous beds; keep impacts &gt;150 ms apart as separate cues. |
| **Output** | Raw cue list (pre-verifier) |

**LOW:** one A/V call over the whole clip.

**HIGH** ([`annotate/proposer.py`](annotate/proposer.py)):

- Split into ~**6 s** windows with **1 s** overlap; edges snap to nearby cuts (±1.5 s).
- Per window: A/V proposer on the subclip, then optional audio-only proposer on the window WAV.
- Merge proposals (audio-only labels preferred when both fire).
- Seed dialogue cues from ASR segments so speech is not missed.

Cue fields the model must return: `cue_id`, `anchor_id`, `audio_type`, `audio_class`, `cue_trigger`, `alignment_relation`, `event_type`, `start_time_ms`, `duration_ms`, `end_time_ms`, `visual_evidence`, `source_audio_context`, optional `impact_times_ms` / `transcript`.

Vocabularies:

- `audio_type` ∈ `SFX`, `AMBIENCE`, `MUSIC`, `DIALOGUE`, `MOVIE_BGM` (+ `NARRATOR` accepted downstream)
- `event_type` ∈ `INSTANT`, `EVENT`, `CONTINUOUS`, `SEGMENT`, `GLOBAL`
- `alignment_relation` ∈ `SYNCHRONOUS`, `ANTICIPATORY`, `REACTIVE`, `CONTEXTUAL`, `TRANSITIONAL`

### 3. Rule verifier + snap (always; no LLM)

[`annotate/verifier.py`](annotate/verifier.py) + [`annotate/snap.py`](annotate/snap.py):

1. **Audibility** — mark cues with/without a nearby onset/speech within `audibility_tol_ms` (150 ms)
2. **Split** — explode multi-onset blobs denser than `min_impact_gap_ms` (150 ms)
3. **Merge** — near-duplicate cues
4. **Snap** (DSP owns final times):
   - SFX → nearest `effects` then `mixture` `audio_onset` within `sfx_snap_ms` (250 ms)
   - DIALOGUE → nearest `speech_start`
   - AMBIENCE/MUSIC → nearest `cut` / `music_boundary` within `bed_snap_ms` (600 ms)
5. Soft-reject SFX/DIALOGUE marked `not_audible` (unless ASR-sourced)

Rejected cues are kept in `rejected_cues.json` / `verifier_log.jsonl` — never silently dropped.

### 4. Judge agent (HIGH only)

| | |
| --- | --- |
| **Model** | `models.judge` → `gemini-3.1-pro-preview` |
| **Prompt** | `JUDGE_PROMPT` + proposed cues + truncated anchors + visual story |
| **Job** | Per cue: `keep` / `drop` / `relabel`. **Must not invent new `start_time_ms`.** |
| **Output** | Filtered/relabeled cue list |

---

## Per-clip artifacts (schema 2.0.0)

```text
features/<video_id>/<clip_id>/
  video_tokens.pt
  token_timestamps.json
  video_token_metadata.json
  perception/
    anchors.json              # Layer 0 timeline (shared)
    stems/                    # HIGH Demucs WAVs when saved
  annotations/<low|high>/
    gemini_explanation.json
    cue_annotations.json      # cues + rejected_cues + verifier_log
    rejected_cues.json
    verifier_log.jsonl
    llm_calls.jsonl
    llm/                      # raw Gemini responses per stage
    pipeline.log              # short stage lines
    pipeline.ui.log           # structured JSONL for tools/view_ui_log
    record.json
```

Provenance on cues: `source: "gemini+dsp"`, `annotation_type: "pseudo_ground_truth"`, `human_verified: false`. Timing after snap records `snap_delta_ms` + `anchor_id`.

Inspect / compare:

```bash
cd dataset_generating_pipepline
python -m tools.inspect_clip --features-dir ../Dataset/extracted/features \
  --video-id g-6g2uEjF1s --clip-id clip_001 --level high
python -m tools.compare_levels --features-dir ../Dataset/extracted/features \
  --video-id g-6g2uEjF1s --clip-id clip_001
```

---

## Clip segmentation (acquisition)

Default acquisition uses **scene-grouped** clips (PySceneDetect → pack toward 30 s, absorb tails &lt; 12 s). Pass `--segmentation fixed` for contiguous 30 s slices.

A 125-second source with scene cuts yields roughly four ~30 s clips whose boundaries land on cuts, not arbitrary seconds.

---

## Installation

Use Python 3.12 and this repository's `requirements.txt`. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install FFmpeg and FFprobe and ensure both are on `PATH`. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the process environment or in `.env` at the repository root (see `.env.example`). VideoMAE model weights are downloaded from Hugging Face on first use; `--device auto` selects CUDA when available and CPU otherwise. HIGH level also needs Demucs + faster-whisper for full fidelity (graceful fallbacks if missing).

---

## Acquire a list of YouTube links

Put one URL per row in `Dataset/metadata/youtube_links.csv`. Leave `status` empty for a new link. A row whose `status` is `processed` is skipped. From this repository root:

```bash
python run.py
```

Or add a link and run it in one step:

```bash
python run.py --url "https://www.youtube.com/watch?v=XXXX"
```

Downloaded sources and clips land under `Dataset/raw_videos/<youtube_id>/`. The same clips are linked into `Dataset/videos/<youtube_id>_clip0.mp4`, … for grounding loaders. `Dataset/metadata/clip_index.csv` lists those files and source intervals.

After download, the same command runs feature extraction + cue annotation into `Dataset/extracted/features/<youtube_id>/<clip_id>/`, and merges cues into `Dataset/metadata/queries.json` as `label_source=gemini_pseudo` records named `<youtube_id>_clip0`.

- Empty `extract_status` → extract next run  
- `extract_status=extracted` → skipped  
- Download failure → `status=download_error` (run continues)  
- `--force` reprocesses `processed` + `extracted` rows  

```bash
python run.py --self-test
```

---

## Acquire one video

```bash
python dataset_generating_pipepline/download_and_clip.py --url "https://www.youtube.com/watch?v=XXXX"
```

Optional: `--output-dir Dataset/raw_videos`, `--clip-duration 30`, `--force`.

```text
Dataset/raw_videos/<video_id>/
  source.mp4
  clip_000.mp4
  …
  video_metadata.json
  processing_status.json
```

Acquisition fails early if the source has no audio. `processing_status.json` is the builder resume ledger.

---

## Build the dataset

```bash
python dataset_generating_pipepline/build_dataset.py --input-dir Dataset/raw_videos --output-dir Dataset/extracted
```

Useful filters:

```text
--video-id <youtube_id>
--clip-id clip_003
--processing-level low|high
--force
--device auto|cpu|cuda
--video-model MCG-NJU/videomae-base
--gemini-model gemini-3-flash-preview
--max-retries 3
--log-level INFO|DEBUG|WARNING|ERROR
```

Per clip the builder: validates A/V → VideoMAE tokens → Layer-0 anchors → Gemini annotate (see above) → atomic `record.json` + JSONL row. Gemini always receives the original unmuted media; no silent derivative is created. Feature/Gemini caches include the clip SHA-256 so replacing a clip invalidates stale artifacts.

Dataset tree:

```text
Dataset/extracted/
  dataset.jsonl
  dataset_metadata.json
  splits.json
  features/<video_id>/<clip_id>/…
```

Each JSONL record references the `[T,D]` VideoMAE tensor rather than duplicating it per cue. Splits are deterministic SHA-256 buckets by YouTube video ID (all clips of one source stay together). The per-video ledger skips valid `completed` entries, retries failures, and recovers stale `processing` after lock acquisition. Expensive stages cache separately so a failed later stage does not redo earlier work. `--force` recomputes selected clips without deleting the dataset tree.

---

## Resume and retry

Interrupt with Ctrl+C at any time. The active clip returns to `pending`; already-written feature, perception, and Gemini artifacts are validated and reused. Exceptions mark that clip `failed` and allow subsequent clips to continue. Nonzero exit code means one or more clip failures.

---

## Module map

| Path | Role |
| --- | --- |
| `build_dataset.py` | Clip loop: validate → VideoMAE → anchors → annotate → record |
| `download_and_clip.py` / `segment_clips.py` | YouTube acquire + scene/fixed segmentation |
| `perception/timebase.py` | A/V start delta, VFR, PTS maps |
| `perception/onsets.py` | v1 mel flux onset detection |
| `perception/onsets_v2.py` | v2 SuperFlux + refine + offsets |
| `perception/sed/` | Frame-level SED backends + gating |
| `perception/video_cues.py` / `avsync.py` | Motion/cut cues + A/V offset estimate |
| `perception/scenes.py` | Legacy visual cuts (+ PySceneDetect for segmentation) |
| `perception/separation.py` | Demucs stems (HIGH) |
| `perception/speech.py` | Whisper ASR → speech_start |
| `perception/beats.py` / `tagger.py` | Beat grid + audio tags |
| `perception/anchors.py` | Merge Layer 0 → `anchors.json` (v1/v2) |
| `annotate/prompts.py` | Explanation / proposer / judge prompts |
| `annotate/proposer.py` | Windowing + slot-fill Gemini calls |
| `annotate/snap.py` | Snap v1 nearest / v2 Hungarian |
| `annotate/verifier.py` | Rules + optional judge |
| `annotate/orchestrator.py` | Explanation → propose → verify → write + LLM logging |
| `eval/` | Gold I/O, metrics, run_eval, ablate |
| `tools/plot_anchors.py` | Debug PNG for anchors + cues |
| `tools/diagnose_anchors.py` | **Per-component** anchor noise test (pass any video path) |
| `tests/test_anchor_components.ipynb` | Interactive notebook: set `VIDEO_PATH`, run each detector cell |
| `docs/ANCHOR_COMPONENTS.md` | Source-file map + how to read diagnostics |
| `ui_log.py` / `tools/view_ui_log.py` | Structured run logs |
| `pipeline_config.json` | Levels, models, windows, snap, perception knobs |
