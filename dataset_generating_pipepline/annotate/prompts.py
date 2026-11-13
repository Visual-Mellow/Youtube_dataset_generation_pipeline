"""Dataset-annotation prompts (audible-source framing).

These are intentionally separate from the production sound-design prompts in
``Utils/prompts.py``. The dataset describes what is *audible in the source*,
not what a Foley artist would invent. Timing comes from Layer-0 anchors;
Gemini only *names* events.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Explanation (one call per clip, both LOW and HIGH)
# ---------------------------------------------------------------------------

EXPLANATION_PROMPT = """\
You are a senior cinematic sound supervisor watching the ATTACHED VIDEO with
its ORIGINAL AUDIO. The audio track is the ground truth for what sounds exist.
Visuals explain *why* a sound is there; they must NOT invent sounds that are
not audible.

Client duration hint (ms, may be approximate): {video_duration_ms}
Director's creative intent (optional): {director_prompt}

Watch and LISTEN carefully. Distinguish continuous beds (music, ambience,
engines) from one-shot impacts (hits, pops, door slams). Note dialogue.

Return FOUR fields as JSON ONLY (no markdown):

1) video_content
A timestamped beat sheet covering the FULL clip. Use millisecond ranges, e.g.:
[0–2400ms] ...
For each beat: who/what is on screen, key audible events, continuous sounds,
mood shifts. Keep under ~2500 chars.

2) scene_gravity
WHY this clip matters narratively. Prefer one clear role.

3) scene_interpretation
Cinematic reading: mood, subtext, visual atmosphere, pacing. 3–6 sentences.

4) audio_asthetics
Describe the *actual* sonic world you hear: instrumentation/texture of music,
ambience character, mix priorities. 3–6 sentences.

Also include video_type: exactly one of
ACTION, DIALOGUE, ESTABLISHING, CHASE, FIGHT, REVEAL, EMOTIONAL,
HORROR_TENSION, TRAVEL, OTHER.

JSON schema:
{{
  "video_content": "<timestamped beat sheet>",
  "scene_gravity": "<narrative role>",
  "scene_interpretation": "<cinematic interpretation>",
  "audio_asthetics": "<what you actually hear>",
  "video_type": "<one of the enum values>"
}}
"""

# ---------------------------------------------------------------------------
# Slot-filling cue proposer (LOW: whole clip; HIGH: per window)
# ---------------------------------------------------------------------------

PROPOSER_PROMPT = """\
You are annotating AUDIBLE events in the attached audiovisual clip for a
beat-sync dataset. The audio track is ground truth. Do NOT invent sounds that
are not audible (e.g. a visible helicopter with no rotor noise = no cue).

Clip / window clock:
- Full clip duration (ms): {video_duration_ms}
- This window on the ORIGINAL timeline: [{window_start_ms}, {window_end_ms}]
- Attached media starts at ORIGINAL ms = {clip_origin_ms}
- Attached media duration (ms): {clip_duration_ms}
Every start_time_ms you output MUST be on the ORIGINAL timeline
(window start ≈ {window_start_ms}, NOT 0).

Visual-story context (compact; audio is primary):
{visual_story}

Detected Layer-0 events in this window (anchor times are authoritative —
do NOT invent or shift times; SED hints are suggestions you may override
if the audio disagrees):
{anchors}

TASK — slot-filling:
1. For EACH detected audio_onset / speech_start / sed_event above, either:
   - name the audible sound (audio_class) and set anchor_id, OR
   - say audio_class = "none" if it is just music beat / noise / not a useful cue.
2. Prefer outputting anchor_id and letting DSP own start/end. You MAY add
   events we missed only if confident within ±200 ms; still prefer a nearby
   listed anchor_id when possible.
3. Also emit continuous beds (AMBIENCE / MUSIC) that span meaningful regions
   of this window, using music_boundary / cut anchors when helpful.
4. Dialogue: emit DIALOGUE cues using speech_start transcripts when present.
5. Visual context: intensity_change anchors mark real picture changes
   (|mean pixel Δ|≥5). Use them as visual_evidence / cue_trigger grounding;
   do NOT invent audio from intensity alone — only name sounds that are audible.

GRANULARITY (critical for beat-sync):
- Impacts more than ~150 ms apart = SEPARATE INSTANT/EVENT cues
  (punch @0.5s, kick @1.0s, gunshot @1.5s — NOT one "fight scene").
- Impacts denser than 150 ms = one CONTINUOUS cue with impact_times_ms listing
  every hit time inside the span.
- Do NOT merge distinct hits into one long blob.

DESCRIPTION (audio_class):
- Concrete caption of what is audible: source, material, mechanism, distance.
- NO negation clauses ("NOT birds"). Those belong in a later generator prompt.
- For continuous beds prefix with "CONTINUOUS: ".

audio_type ∈ {{SFX, AMBIENCE, MUSIC, DIALOGUE, MOVIE_BGM}}
event_type ∈ {{INSTANT, EVENT, CONTINUOUS, SEGMENT, GLOBAL}}
alignment_relation ∈ {{SYNCHRONOUS, ANTICIPATORY, REACTIVE, CONTEXTUAL, TRANSITIONAL}}

Return JSON ONLY:
{{
  "cues": [
    {{
      "cue_id": "cue_001",
      "anchor_id": "a041",
      "audio_type": "SFX",
      "audio_class": "bottle cap pried off with short carbonation hiss",
      "cue_trigger": "cashier opens second bottle",
      "alignment_relation": "SYNCHRONOUS",
      "event_type": "INSTANT",
      "start_time_ms": 21190,
      "duration_ms": 400,
      "end_time_ms": 21590,
      "visual_evidence": "hands on bottle at counter",
      "source_audio_context": "clear pop + hiss in mix",
      "impact_times_ms": [],
      "transcript": null
    }}
  ]
}}
Omit cues whose audio_class is "none". Clamp all times to the window and to
the clip duration. Prefer lean, precise lists over sparse blobs.
"""

AUDIO_ONLY_PROPOSER_PROMPT = """\
You are annotating AUDIBLE events from AUDIO ONLY (no video). The attached
media is the audio track of a window. Invent nothing that is not audible.

Clock (ORIGINAL timeline):
- Full clip duration (ms): {video_duration_ms}
- Window: [{window_start_ms}, {window_end_ms}]
- Attached audio starts at ORIGINAL ms = {clip_origin_ms}
- Attached audio duration (ms): {clip_duration_ms}

Detected Layer-0 events:
{anchors}

Same slot-filling rules as the A/V pass: name each listed event or "none";
add missed events only with ±200 ms confidence; keep impacts separate when
>150 ms apart; audio_class is a concrete audible description with no negations.

Return JSON ONLY with the same "cues" schema. Use ORIGINAL-timeline times.
"""

# ---------------------------------------------------------------------------
# Judge (HIGH only) — semantics, NO timing authority
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a verifier for a beat-sync audio-event dataset. You may KEEP, DROP, or
RELABEL cues. You must NOT invent new start_time_ms values — timing is owned
by DSP anchors. If a cue's time looks wrong, DROP it or keep it and note the
concern; the snap stage will retime to the nearest anchor.

Full-clip duration (ms): {video_duration_ms}

Visual-story context:
{visual_story}

Layer-0 anchors (truncated):
{anchors}

Proposed cues (after rule-based verifier):
{cues}

For EACH cue return a verdict:
- verdict: "keep" | "drop" | "relabel"
- grounded: true if the sound is actually audible near that time
- reason: short justification
- audio_class: new label ONLY when verdict="relabel"; else omit
- drop when: not audible, pure visual hallucination, or duplicate of a better cue
- Extra scrutiny: cues with uncertainty_ms > 60 need stronger audible evidence
  to KEEP; prefer DROP when grounding is weak.

Return JSON ONLY:
{{
  "verdicts": [
    {{"id": "cue_001", "verdict": "keep", "grounded": true, "reason": "clear bottle pop at onset"}},
    {{"id": "cue_009", "verdict": "drop", "grounded": false, "reason": "helicopter visible but no rotor in audio"}},
    {{"id": "cue_012", "verdict": "relabel", "grounded": true, "audio_class": "glass bottle clink on laminate", "reason": "tagger+audio agree on glass, not wood"}}
  ]
}}
"""
