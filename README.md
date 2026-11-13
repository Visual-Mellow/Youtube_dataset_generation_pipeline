# YouTube dataset pipeline

This repository downloads YouTube videos, cuts them into clips, and writes VideoMAE features plus Gemini cue labels (timing owned by local DSP). Paste a link, then run one command.

Generated videos, clip files, and metadata stay on your machine. The only dataset file in git is `Dataset/metadata/youtube_links.csv`.

## What you need

- Python 3.12
- [FFmpeg](https://ffmpeg.org/) on your `PATH` (`ffmpeg` and `ffprobe`)
- A Gemini API key

On macOS:

```bash
brew install ffmpeg
```

On Debian or Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Setup

From this repository (fresh clone included):

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Creates Dataset/ folders (raw_videos, videos, clips, extracted, metadata),
# seed CSVs / queries.json, copies .env.example → .env, and installs deps.
python setup_workspace.py --install --check
```

Or step by step:

```bash
python setup_workspace.py          # folders + seed files only
python -m pip install -r requirements.txt
```

Open `.env` and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

`setup_workspace.py` is safe to re-run: it will not overwrite an existing `.env` or a non-empty `youtube_links.csv`.

## Run

**Option A.** Paste a link on the command line. It is saved into `Dataset/metadata/youtube_links.csv`, then downloaded and extracted:

```bash
python run.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Option B.** Edit the links file yourself. Put one URL in the `youtube_url` column and leave `status` empty:

```text
youtube_url,video_id,status,error,n_clips,clips_dir,processed_at,extract_status,extract_error,processing_level
https://www.youtube.com/watch?v=VIDEO_ID,,,,,,,,
```

Then:

```bash
python run.py
```

Rows already marked `processed` and `extracted` are skipped. Run the same command again to retry `download_error` and `extract_error` rows. Pass `--force` to redo finished rows. Use `--processing-level high` (or the CSV `processing_level` column) for Demucs + multi-window proposers + judge.

```bash
python run.py --force
python run.py --self-test
python run.py --processing-level low
```

## Package layout

```text
Youtube_dataset_generation_pipeline/
├── run.py                          # CLI entry → dataset_generating_pipepline/run_youtube_links.py
├── requirements.txt
├── Dataset/                        # runtime data (mostly gitignored)
│   ├── metadata/youtube_links.csv
│   ├── raw_videos/<youtube_id>/
│   ├── videos/
│   └── extracted/
└── dataset_generating_pipepline/   # pipeline package
    ├── run_youtube_links.py        # batch loop over CSV
    ├── orchestrator.py             # download → publish clips → extract
    ├── build_dataset.py            # per-clip VideoMAE + perception + annotate
    ├── core/                       # shared config & schemas
    │   ├── constants.py            # Dataset paths, CSV columns
    │   ├── schemas.py              # PipelineConfig dataclasses
    │   ├── config.py               # load_pipeline_config()
    │   └── pipeline_config.json
    ├── video_helpers/              # acquire + cut media
    │   ├── download_and_clip.py
    │   ├── segment_clips.py
    │   ├── split_scenes.py         # standalone flat-MP4 helper
    │   └── video_extraction.py     # wraps build_dataset + queries export
    ├── perception/                 # Layer-0 DSP anchors
    │   ├── anchors.py
    │   ├── onsets_v2.py, video_cues.py, timebase.py, …
    │   └── sed/
    ├── annotate/                   # Gemini explain / propose / verify / judge
    ├── utils/                      # CSV I/O, UI logs, logger
    ├── tools/                      # diagnose_anchors, view_ui_log, …
    ├── eval/                       # gold labels + metrics
    ├── tests/
    └── docs/
```

## Pipeline architecture

```text
run.py
  └─ run_youtube_links.py
       └─ orchestrator.process_links()
            ├─ video_helpers.download_and_clip.acquire_url()
            │     download MP4 → segment_clips (scene|fixed) → raw_videos/<id>/clip_*.mp4
            ├─ publish_clips() → Dataset/videos/<id>_clipN.mp4 + clip_index.csv
            └─ video_helpers.video_extraction.extract_video()
                  └─ build_dataset.main()
                        per clip:
                          VideoMAE tokens
                          perception (Layer-0 anchors)
                          annotate (LOW|HIGH Gemini agents + DSP snap)
                  └─ export_grounding_queries() → Dataset/metadata/queries.json
```

Config lives in `dataset_generating_pipepline/core/pipeline_config.json`. LOW vs HIGH flags, anchor DSP, and prompts are documented in [dataset_generating_pipepline/README_DSP_DATASET.md](dataset_generating_pipepline/README_DSP_DATASET.md).

## Where the data goes

| Path | What it is |
| --- | --- |
| `Dataset/metadata/youtube_links.csv` | Link list and per-video status. Only dataset file kept in git. |
| `Dataset/raw_videos/<youtube_id>/` | Source video, clips, `video_metadata.json`, `processing_status.json`. |
| `Dataset/videos/` | Published clips named `<youtube_id>_clip0.mp4`, … |
| `Dataset/metadata/clip_index.csv` | Clip filenames and source time ranges. |
| `Dataset/metadata/queries.json` | Gemini cues merged for grounding. |
| `Dataset/extracted/dataset.jsonl` | One JSON record per clip. |
| `Dataset/extracted/features/<youtube_id>/<clip_id>/` | VideoMAE tokens, perception anchors, annotations. |

`Dataset/extracted/`, `Dataset/raw_videos/`, `Dataset/videos/`, and every metadata file except `youtube_links.csv` are gitignored.

## One video without the CSV

```bash
python dataset_generating_pipepline/video_helpers/download_and_clip.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
python dataset_generating_pipepline/build_dataset.py --video-id VIDEO_ID --processing-level low
```

Useful builder flags: `--clip-id clip_000`, `--force`, `--device auto|cpu|cuda`, `--video-model MCG-NJU/videomae-base`, `--gemini-model gemini-3-flash-preview`, `--anchor-version 1|2`.

## What a run does

Default segmentation is scene-aware (PySceneDetect grouping toward ~30 s clips; `segmentation.mode=fixed` keeps contiguous fixed-duration slices).

For each new link the pipeline:

1. Downloads an MP4 with audio using `yt-dlp`. A private, removed, or HTTP 403 video is marked `download_error` and the next link continues.
2. Cuts clips with FFmpeg and records `video_metadata.json` and `processing_status.json`.
3. Extracts VideoMAE tokens, Layer-0 DSP anchors, a Gemini explanation, and cue timestamps (snapped to DSP).
4. Appends cues to `Dataset/metadata/queries.json`.

Stop with Ctrl+C at any time. Finished clips are reused on the next run. Cue labels are Gemini pseudo-labels (`human_verified: false`), not human ground truth.

VideoMAE weights download from Hugging Face the first time a clip is extracted. `--device auto` uses CUDA when available, otherwise CPU.

## UI pipeline logs

| Path | What it is |
| --- | --- |
| `Dataset/extracted/features/<video_id>/<clip_id>/annotations/<level>/pipeline.ui.log` | Per-clip run |
| `Dataset/metadata/youtube_links.ui.log` | Batch `run.py` summary |
| `pipeline.log` | Plain-text timeline |

```bash
python dataset_generating_pipepline/tools/view_ui_log.py \
  Dataset/extracted/features/VIDEO_ID/clip_000/annotations/low/pipeline.ui.log \
  --open
```
