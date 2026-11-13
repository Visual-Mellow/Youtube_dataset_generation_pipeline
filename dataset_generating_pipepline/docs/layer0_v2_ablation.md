# Layer-0 v2 ablation notes

## Defaults locked in `pipeline_config.json`

| Knob | Value | Rationale |
|---|---|---|
| `anchor_version` | 2 | SuperFlux + refine + optional SED |
| `onsets_v2.mad_k` | 3.5 | Adaptive median+MAD; stable on synthetic clicks |
| `onsets_v2.min_gap_ms` | 35 | Dense footsteps/gunfire; was 100 in v1 |
| `onsets_v2.cross_band_merge_ms` | 25 | Merge HF/mid co-firings |
| `refine.floor_db` / `min_snr_db` | 6 | Rise-start backtrack; low-SNR flagged |
| `sed.on/off` | 0.5 / 0.3 | Hysteresis; no long median filter |
| `fusion` weights | 0.35/0.30/0.15/0.15/0.05 | onset/sed/snr/visual/beat |
| `snap.version` | 2 | Hungarian class-aware assignment |

## Synthetic suite (gate for M2)

Run:

```bash
cd dataset_generating_pipepline
python -m tests.test_synthetic
python -m eval.ablate --synthetic-only
```

Acceptance targets:

- Clean signals: median onset error ≤ 10 ms (plan: ≤ 5 ms aspirational)
- 10 dB SNR: median ≤ 25 ms

## Gold-set protocol

1. `python -m eval.make_gold_template --features-dir … --video-id … --clip-id …`
2. Hand-edit Audacity labels (rename machine rows to `GOLD|…` or add new rows)
3. `python -m eval.import_audacity --labels … --clip-id video/clip --out eval/gold/….json`
4. `python -m eval.run_eval --features-dir … --anchor-version 1` then `--anchor-version 2`

Record baseline JSON under `eval/results/`. Re-run ablations after ≥20 labeled clips and write winning knobs back into `pipeline_config.json`.

## A/B via `anchor_version`

```bash
python build_dataset.py --video-id … --clip-id … --processing-level low --anchor-version 1 --force
python build_dataset.py --video-id … --clip-id … --processing-level low --anchor-version 2 --force
```

Cache keys include `anchor_version`, `onsets_v2`, `sed_backend`, and `separation_used`.
