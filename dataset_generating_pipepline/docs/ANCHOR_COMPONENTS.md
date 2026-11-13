# Anchor components — source map + diagnostic

Use this when cues look redundant: isolate which Layer-0 detector is noisy
before changing Gemini prompts.

## Interactive notebook (recommended)

Open `tests/test_anchor_components.ipynb`:

1. Set `VIDEO_PATH` in the setup cell.
2. Run setup once (loads audio).
3. Run any component cell on its own — each prints density + a timeline plot.
4. Run **Noise ranking** at the bottom to compare what you executed.

## CLI (same runners)

```bash
cd dataset_generating_pipepline

# List components + source files
python -m tools.diagnose_anchors --list

# Full isolation run
python -m tools.diagnose_anchors \
  --video ../Dataset/raw_videos/nq1M_Wc4FIc/clip_000.mp4

# Only the usual noise suspects
python -m tools.diagnose_anchors \
  --video /absolute/path/to/clip.mp4 \
  --only onsets_v2,onsets_v2_effects,intensity_change,video_cues,scenes

# A/B onset sensitivity
python -m tools.diagnose_anchors --video clip.mp4 --only onsets_v2 --mad-k 6.5

# Skip slow / optional deps
python -m tools.diagnose_anchors --video clip.mp4 --skip sed,speech,fusion,onsets_v2_effects
```

Outputs (next to the video by default: `<stem>_anchor_diag/`):

| File | Purpose |
|---|---|
| `summary.txt` / `summary.json` | Noise ranking across components |
| `<component>.json` | Full event list for that detector |
| `<component>.txt` | Audacity label track — load onto the clip audio |

## Source files

| Component | File | What it emits |
|---|---|---|
| `onsets_v1` | `perception/onsets.py` | Classic mel-flux `audio_onset` |
| `onsets_v2` | `perception/onsets_v2.py` | SuperFlux + refine (often densest) |
| `onsets_v2_effects` | `onsets_v2.py` + `separation.py` | Same, Demucs effects stem only |
| `intensity_change` | `perception/video_cues.py` | Mean pixel `\|Δ\| ≥ min_change` |
| `video_cues` | `perception/video_cues.py` | cut / motion_* / flash / intensity |
| `scenes` | `perception/scenes.py` | ffmpeg scene_score cut/motion |
| `speech` | `perception/speech.py` | Whisper `speech_start` |
| `sed` | `perception/sed/*` | Class-aware `sed_event` |
| `beats` | `perception/beats.py` | Beat grid (support, not SFX) |
| `fusion` | `perception/anchors.py` | Merged timeline → `anchors.json` |

Config knobs: `pipeline_config.json` → `perception.onsets_v2`, `perception.video_cues_v2`, `perception.sed`, `perception.fusion`.

## How to read noise

1. Open the clip in Audacity, import each `.txt` as a label track.
2. The track with the densest ticks (high `n`, low `median_gap_ms`, high `noise_score` in `summary.txt`) is the first place to tune.
3. Re-run with `--mad-k` / `--min-intensity-change` until density looks usable (~1–4 events/sec for SFX clocks is a good ballpark for 30s action clips).
