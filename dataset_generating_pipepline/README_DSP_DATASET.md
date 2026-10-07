# Dataset pipeline reference

Quick start, setup, and the one-command run live in the repository [README](../README.md). This file describes clip behavior and the artifacts each stage writes.

The acquisition and builder commands below operate on fixed-duration, sequential, **non-overlapping** clips. The default clip duration is 30 seconds. A 125-second source yields `0–30`, `30–60`, `60–90`, `90–120`, and `120–125` second clips. The final clip may be shorter; the extraction ledger records each half-open source interval in milliseconds.

## Installation

Use Python 3.12 and this repository's `requirements.txt`. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install FFmpeg and FFprobe and ensure both are on `PATH`. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the process environment or in `.env` at the repository root (see `.env.example`). VideoMAE model weights are downloaded from Hugging Face on first use; `--device auto` selects CUDA when available and CPU otherwise.

## Acquire a list of YouTube links

Put one URL per row in `Dataset/metadata/youtube_links.csv`. Leave `status` empty for a new link. A row whose `status` is `processed` is skipped. From this repository root:

```bash
python run.py
```

Or add a link and run it in one step:

```bash
python run.py --url "https://www.youtube.com/watch?v=XXXX"
```

Downloaded sources and the original clips are written under `Dataset/raw_videos/<youtube_id>/`. The same clips are also linked into `Dataset/videos/<youtube_id>_clip0.mp4`, `Dataset/videos/<youtube_id>_clip1.mp4`, and so on, which is the filename the grounding loaders read. `Dataset/metadata/clip_index.csv` lists those files and their source intervals.

Once the clips exist, the same command runs cue extraction: VideoMAE tokens, a Gemini explanation, and Gemini cue timestamps. Those artifacts land in `Dataset/extracted/features/<youtube_id>/<clip_id>/`, and the cues are merged into `Dataset/metadata/queries.json` as `label_source=gemini_pseudo` records named `<youtube_id>_clip0`. A downloaded row whose `extract_status` is empty is extracted on the next run. `extract_status=extracted` is skipped. An extraction failure is stored in `extract_error` and the next video continues.

If a video cannot be downloaded (HTTP 403, private, removed), the row is set to `status=download_error`, `error` explains that the video is not downloadable, and the next row is processed. The run does not stop on that row. Run the same command again to retry `download_error` and `extract_error` rows. Rows that are both `processed` and `extracted` stay skipped unless you pass `--force`.

```bash
python run.py --self-test
```

## Acquire one video

```bash
python dataset_generating_pipepline/download_and_clip.py --url "https://www.youtube.com/watch?v=XXXX"
```

Optional arguments:

```text
--output-dir Dataset/raw_videos
--clip-duration 30
--force
```

`yt-dlp` downloads/merges an MP4-compatible source. A valid existing `source.mp4` is reused; `--force` authorizes replacing it. FFmpeg extracts each interval while keeping its audio stream. There is no overlap option and no scene detection in primary segmentation.

```text
Dataset/raw_videos/<video_id>/
  source.mp4
  clip_000.mp4
  clip_001.mp4
  video_metadata.json
  processing_status.json
```

`video_metadata.json` includes the source ID, URL/title when available, duration, frame rate, dimensions, clip duration, and each clip's source start/end. Acquisition fails early if the downloaded source has no audio stream. `processing_status.json` is the builder's resume ledger.

## Build the dataset

```bash
python dataset_generating_pipepline/build_dataset.py --input-dir Dataset/raw_videos --output-dir Dataset/extracted
```

Useful filters and settings:

```text
--video-id <youtube_id>
--clip-id clip_003
--force
--device auto|cpu|cuda
--video-model MCG-NJU/videomae-base
--gemini-model gemini-3-flash-preview
--max-retries 3
--log-level INFO|DEBUG|WARNING|ERROR
```

The builder validates each original audiovisual clip, extracts pretrained VideoMAE temporal tokens, generates a Gemini explanation and cue annotations, validates timestamps and schema, and writes an atomic per-clip record and JSONL row. Gemini receives the original unmuted clip on both calls. No silent derivative is created. Feature and Gemini caches include the source clip SHA-256, so replacing a clip invalidates stale artifacts. The dataset prompt uses source audio as context while asking for a visually grounded cinematic soundscape rather than a transcription of the soundtrack.

At the default `INFO` level, the builder prints a short success line when each clip's tokens, Gemini explanation, and cue annotations are generated or reused. Fully skipped clips do not produce per-clip progress lines; the final summary reports skipped video and clip counts. Gemini SDK and HTTP request chatter is hidden; use `--log-level DEBUG` for lower-level details.

```text
Dataset/extracted/
  dataset.jsonl
  dataset_metadata.json
  splits.json
  features/<video_id>/<clip_id>/
    video_tokens.pt
    token_timestamps.json
    video_token_metadata.json
    gemini_explanation.json
    cue_annotations.json
    record.json
```

Each JSONL record represents one clip and references its `[T,D]` tensor rather than duplicating it per cue. Tokens store actual extracted shape; VideoMAE base is expected to have `D=768`, but the builder records the installed model's actual hidden size and `T`. Timestamp entries map pooled temporal tubelets to approximate clip-relative intervals based on sampled input frame groups.

All generated cue and timing labels are marked `source: "gemini"`, `annotation_type: "pseudo_ground_truth"`, and `human_verified: false` at dataset provenance level. They are not human ground truth. Cue types follow the production ontology; `event_type` and `alignment_relation` are dataset-level controlled vocabularies. Timestamp normalization/clamping is recorded per cue.

The dataset-only `video_type` vocabulary is `ACTION`, `DIALOGUE`, `ESTABLISHING`, `CHASE`, `FIGHT`, `REVEAL`, `EMOTIONAL`, `HORROR_TENSION`, `TRAVEL`, and `OTHER`; Gemini supplies the class and the record stores `video_type_source: "gemini"`. `event_type` uses `INSTANT`, `EVENT`, `CONTINUOUS`, `SEGMENT`, or `GLOBAL`. `alignment_relation` uses `SYNCHRONOUS`, `ANTICIPATORY`, `REACTIVE`, `CONTEXTUAL`, or `TRANSITIONAL`.

Splits are deterministic SHA-256 buckets by original YouTube video ID, keeping all clips from one source together. The per-video processing ledger skips valid `completed` entries, processes pending entries, retries failures on later runs, and recovers stale `processing` entries after acquiring the single-builder lock. Expensive stages cache to separate artifacts, so a failed later stage does not require rerunning earlier successful stages. `--force` recomputes the selected clips without deleting the dataset tree.

## Resume and retry

Interrupt with Ctrl+C at any time. The active clip is returned to `pending`; already-written feature and Gemini artifacts are validated and reused on the next invocation. Exceptions mark that clip `failed`, record details, and allow subsequent clips to continue. A nonzero exit code indicates one or more clip failures.
