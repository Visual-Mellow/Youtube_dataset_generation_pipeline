# YouTube dataset pipeline

This repository downloads YouTube videos, cuts them into non-overlapping clips, and writes VideoMAE features plus Gemini cue labels. It runs on its own. Paste a link, then run one command.

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

From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

## Run

**Option A.** Paste a link on the command line. It is saved into `Dataset/metadata/youtube_links.csv`, then downloaded and extracted:

```bash
python run.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Option B.** Edit the links file yourself. Put one URL in the `youtube_url` column and leave `status` empty:

```text
youtube_url,video_id,status,error,n_clips,clips_dir,processed_at,extract_status,extract_error
https://www.youtube.com/watch?v=VIDEO_ID,,,,,,,
```

Then:

```bash
python run.py
```

Rows already marked `processed` and `extracted` are skipped. Run the same command again to retry `download_error` and `extract_error` rows. Pass `--force` to redo finished rows.

```bash
python run.py --force
python run.py --self-test
```

## Where the data goes

| Path | What it is |
| --- | --- |
| `Dataset/metadata/youtube_links.csv` | Your link list and per-video status. This is the only dataset file kept in git. |
| `Dataset/raw_videos/<youtube_id>/` | Source video, 30-second clips, and clip metadata. |
| `Dataset/videos/` | Copies of those clips named `<youtube_id>_clip0.mp4`, `<youtube_id>_clip1.mp4`, and so on. |
| `Dataset/metadata/clip_index.csv` | Clip filenames and source time ranges. |
| `Dataset/metadata/queries.json` | Gemini cues merged for grounding. |
| `Dataset/extracted/dataset.jsonl` | One JSON record per clip. |
| `Dataset/extracted/features/<youtube_id>/<clip_id>/` | VideoMAE tokens, Gemini explanation, and cue annotations. |

`Dataset/extracted/`, `Dataset/raw_videos/`, `Dataset/videos/`, and every metadata file except `youtube_links.csv` are gitignored.

## One video without the CSV

```bash
python dataset_generating_pipepline/download_and_clip.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
python dataset_generating_pipepline/build_dataset.py --video-id VIDEO_ID
```

Clips land in `Dataset/raw_videos/`. Features land in `Dataset/extracted/`.

Useful builder flags: `--clip-id clip_000`, `--force`, `--device auto|cpu|cuda`, `--video-model MCG-NJU/videomae-base`, `--gemini-model gemini-3-flash-preview`.

## What a run does

The default clip length is 30 seconds, sequential and non-overlapping. A 125-second video becomes `0–30`, `30–60`, `60–90`, `90–120`, and `120–125`. The last clip can be shorter.

For each new link the pipeline:

1. Downloads an MP4 with audio using `yt-dlp`. A private, removed, or HTTP 403 video is marked `download_error` and the next link continues.
2. Cuts contiguous clips with FFmpeg and records `video_metadata.json` and `processing_status.json`.
3. Extracts VideoMAE tokens, a Gemini explanation, and Gemini cue timestamps.
4. Appends cues to `Dataset/metadata/queries.json`.

Stop with Ctrl+C at any time. Finished clips are reused on the next run. Cue labels are Gemini pseudo-labels (`human_verified: false`), not human ground truth.

VideoMAE weights download from Hugging Face the first time a clip is extracted. `--device auto` uses CUDA when it is available, otherwise CPU.

More detail on clip records, cue fields, and resume behavior is in [dataset_generating_pipepline/README_DSP_DATASET.md](dataset_generating_pipepline/README_DSP_DATASET.md).
