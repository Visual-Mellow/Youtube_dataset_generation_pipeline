from langchain_core.prompts import PromptTemplate

gemini_audio_prompt = PromptTemplate(
    input_variables=["story_text", "speed_wps", "movie_bgms_csv"],
    template=(
        """Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

Story: "{story_text}"

Reading Speed: {speed_wps} words per second
Total Story Words: Count the words in the story
Total Duration (ms): Calculate as (total_words / {speed_wps}) * 1000

For each sound, you MUST provide:
- audio_class: detailed sound description for SoundGen AI
- audio_type: SFX (short sounds), AMBIENCE (background), or MUSIC (emotional)
- word_index: position (0-based) where the sound should start in the story
- start_time_ms: EXACT start time in milliseconds. Calculate as: (word_index / {speed_wps}) * 1000
- duration_ms: EXACT duration in milliseconds that YOU decide based on:
  * SFX: Decide duration (500-3000ms) based on the specific sound - a single bark might be 800ms, footsteps might be 2000ms, a door slam might be 1200ms
  * AMBIENCE: Decide duration based on story context - how long should this ambience play? Calculate from word_index to where it should end (next scene change, next AMBIENCE, or story end)
  * MUSIC: Decide duration based on emotional arc - how long should this musical element play? Consider the emotional moment and when it should fade (typically 2000-10000ms)
- weight_db: volume adjustment (-10.0 to 5.0, use 6.0 for "loud")

CRITICAL: You MUST provide a specific duration_ms value for EVERY audio cue. Do not leave it to be calculated later. Think about:
- For SFX: How long does this specific sound naturally last?
- For AMBIENCE: When does the scene/environment change in the story?
- For MUSIC: When does the emotional moment peak and fade?

Timing Calculation Rules:
1. start_time_ms = (word_index / {speed_wps}) * 1000 (round to nearest integer)
2. duration_ms = YOUR DECISION based on story context and sound type - provide the exact value
3. For overlapping sounds of the same type (e.g., two AMBIENCE cues), calculate when the first should end (typically when the second starts)
4. Ensure start_time_ms + duration_ms does not exceed total_duration_ms
5. Be precise - your duration_ms values will be used directly without modification

Try keeping as few Audio Cues as possible, not more than 3-4 Audio Cues.

Return ONLY a JSON array with these exact fields:
[
  {{"audio_class": "detailed sound description", "audio_type": "SFX|AMBIENCE|MUSIC", "word_index": 0, "start_time_ms": 0, "duration_ms": 2000, "weight_db": 0.0}}
]

"""
    ),
)

# gemini_audio_prompt_with_narrator_without_movie_bgms = PromptTemplate(
#     input_variables=["story_text", "speed_wps"],
#     template=(
#         """
# You are specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.
# Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

# Story: {story_text}
# Reading Speed: {speed_wps} words per second
# Total Story Words: Count the words in the story
# Cinematic Master Model: Story Analysis & Sync Prompt
# Role & Expertise: You are a Master Sound Designer and Narrative Director Agent. Your task is to perform a deep semantic analysis of the provided story to extract cinematic audio cues and direct a Narrator AI. You must ensure that the audio atmosphere perfectly syncs with the emotional arc and reading pace of the narrator.

# ### 1. NARRATOR AI DIRECTIVES The Narrator AI reads the entire story from word index 0 to the end. You must provide a "Narrator_Style" description that tells the AI exactly how to perform the reading based on the story's genre, mood, and tension.

# Narrator Persona Guidelines: Match the story's context to one of these reference styles or create a custom blend:

# Suspense/Horror: Low pitch, slower pace, breathless or whispering delivery.

# Action/Urgency: Faster pace, higher intensity, clear and sharp articulation.

# Serene/Nature: Calm, moderate pace, smooth intonation with subtle warmth.


# ### 1. NARRATOR DESCRIPTION You must provide a "Narrator_Style" description that tells the AI exactly how to perform the reading based on the story's genre, mood, and tension.

# include the term "very clear audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
# Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech
# The remaining speech features (gender, speaking rate, pitch and reverberation)

# Mini Model - Top 20 Speakers
# Speaker	Similarity Score
# Jon	0.908301
# Lea	0.904785
# Gary	0.903516
# Jenna	0.901807
# Mike	0.885742
# Laura	0.882666
# Lauren	0.878320
# Eileen	0.875635
# Alisa	0.874219
# Karen	0.872363
# Barbara	0.871509
# Carol	0.863623
# Emily	0.854932
# Rose	0.852246
# Will	0.851074

# add description like "A male speaker with a monotone and high-pitched voice is delivering his speech at a really low speed in a confined environment." or "A female speaker with a high-pitched voice is delivering her speech at a really fast speed in a noisy environment." or "Jon speaks with a low-pitched voice is delivering his speech at a really slow speed in a quiet environment." or "Lea speaker with a low-pitched voice is delivering her speech at a really fast speed in a noisy environment."


# ### 2. AUDIO CUE ENGINEERING You must identify any number of critical audio cues that ground the story in a professional soundscape.

# SFX (Short Effects): Punctuate specific actions (e.g., twig snapping, door slam). Duration: 500ms–3000ms.

# AMBIENCE (Environment): Constant background textures (e.g., rain, forest hum). Duration: From the trigger word to the next scene change or end of story.
 
# MUSIC (Emotional Score): Sets the heart of the scene (e.g., "Tense orchestral strings"). Duration: 2000ms–10000ms. This will be generated by model

# MOVIE_BGM (Movie Background Music): Sets the mood of the scene (e.g., "Heroic soundtrack"). Duration: 2000ms–10000ms. This will be retrieved from the movie bgms data. So it might not be excatly match to description or story context.

# ### 3. TIMING & SYNC MATH Precision is mandatory for "Multimodal Alignment".

# word_index: The 0-based position of the word that triggers the sound.

# start_time_ms: The start time of the sound.

# duration_ms: You must provide an exact value.

# For Overlaps: If a new AMBIENCE starts, the previous one of the same type should end at that start_time_ms.

# weight_db: volume adjustment (-15.0 to 6.0). 6.0 = "Defeaning/Loud".

# ### 5. OUTPUT CONSTRAINTS

# JSON ONLY: No conversational filler.

# Keep as many audio cues as you want to cover full story into audio. Focus on story context try to find audio sources, what music or sfx should be included in the story.

# Narrator Object: Include a single narrator_description at the root.

# ### EXAMPLE OF EXPECTED ANALYSIS Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

# JSON

# {{

#   "audio_cues": [
#     {{
#         "story": " The part of the story that the narrator will read with given descrpition , make sure to include pauses and breaks as per the narrator description
        
#         # you might break story into multiple parts and make seprate audio cues for each part
#         ", 
#         "narrator_description": "Tapan speaks at a moderate pace with a low-pitched, gravelly tone to convey mystery. Clear, close-sounding recording with a cold, detached emotional depth",
#         ##### audio que for narrator to know how to read the story
#         "audio_type": "NARRATOR",
#         "start_time_ms": 0,
#         "duration_ms": duration_ms,
#     }},
#     {{
#       "audio_class": "Heavy wooden door creaking open slowly with high-frequency friction",
#       "audio_type": "SFX",
#       "word_index": 1,
#       "start_time_ms": 500,
#       "duration_ms": 1500,
#       "weight_db": 2.0
#     }},
#     {{
#       "audio_class": "Heavy rain hitting glass window with distant thunder rumbles",
#       "audio_type": "AMBIENCE",
#       "word_index": 4,
#       "start_time_ms": 2000,
#       "duration_ms": 8000,
#       "weight_db": -5.0
#     }},
#     {{
#       "audio_class": "Dark cinematic suspense pad with low synth drones",
#       "audio_type": "MUSIC",
#       "word_index": 10,
#       "start_time_ms": 5000,
#       "duration_ms": 10000,
#       "weight_db": 0.0
#     }},
   
#   ]
# }}

# """
#     ),
# )



gemini_audio_prompt_with_narrator_without_movie_bgms = PromptTemplate(
    input_variables=["story_text", "speed_wps"],
    template=(
        """
You are a specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.

Story: {story_text}
Reading Speed: {speed_wps} words per second

Role & Expertise: You are a Master Sound Designer, Narrative Director, and Lead Mixing Engineer. Your task is to extract cinematic audio cues, direct a Narrator AI, and critically balance the volume (weight_db) of all overlapping sounds so the final mix is clear, professional, and not distorted.

### 1. NARRATOR AI DIRECTIVES
The Narrator AI reads the entire story. Provide a "Narrator_Style" description telling the AI how to perform based on genre, mood, and tension.
* Include terms like "very clear audio" or "very noisy audio" for background control.
* Use punctuation (commas) to add small breaks in speech.
* Specify speaker style, gender, speaking rate, pitch, and environment.
* Example Reference Speakers: Jon, Lea, Gary, Jenna, Mike, Laura.
* Example Description: "Jon speaks with a low-pitched voice, delivering his speech at a really slow speed in a quiet environment, pausing carefully for suspense."

### 2. AUDIO CUE ENGINEERING
Identify critical audio cues that ground the story:
* SFX (Short Effects): Punctuate specific actions (e.g., twig snapping). Duration: 500ms–3000ms.
* AMBIENCE (Environment): Constant background textures (e.g., rain). Duration: From trigger to scene change.
* MUSIC (Emotional Score): Sets the heart of the scene. Duration: 2000ms–10000ms. Generated by model.
* MOVIE_BGM: Retrieved atmospheric tracks. Duration: 2000ms–10000ms.

### 3. TIMING & SYNC MATH
Precision is mandatory for Multimodal Alignment.
* word_index: The 0-based position of the word that triggers the sound.
* start_time_ms: The exact start time.
* duration_ms: Exact duration. Overlapping ambiences of the same type must cut off the previous one.

### 4. THE MIX HIERARCHY & AUDIO DUCKING (CRITICAL)
You MUST evaluate all sounds playing at the same `start_time_ms`. Overlapping sounds add up and cause distortion. You must assign `weight_db` using this strict hierarchy:
1. NARRATOR (The Anchor): Always the clearest element. Assumed to be at 0.0 dB.
2. SFX (The Action): Loud but brief. Use -2.0 to -6.0 dB so it doesn't overpower the voice. (Exception: Explosions/Jump scares can peak at 2.0 to 4.0 dB).
3. MUSIC / MOVIE_BGM (The Emotion): Must sit UNDER the narrator. Use -10.0 to -15.0 dB.
4. AMBIENCE (The Void): Must sit at the very bottom of the mix. Use -18.0 to -24.0 dB.

AUDIO DUCKING RULE: If the Narrator is speaking, MUSIC and AMBIENCE must be pushed down to lower volumes (-15 to -20 dB). If the Narrator pauses for a long time, MUSIC can swell up slightly. NEVER put overlapping Ambience, Music, and SFX all above -5.0 dB at the same time.

### 5. OUTPUT CONSTRAINTS
* JSON ONLY. No conversational filler.
* Focus on story context to find what music or sfx should be included.

### EXAMPLE OF EXPECTED ANALYSIS 
Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

JSON
{{
  "audio_cues": [
    {{
        "story": "The door creaked open, ... Rain lashed against the window, ... as he stepped into the cold hall.", 
        "narrator_description": "Gary speaks at a moderate pace with a low-pitched, gravelly tone. Very clear audio in a quiet environment.",
        "audio_type": "NARRATOR",
        "start_time_ms": 0,
        "duration_ms": 7500
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -4.0
    }},
    {{
      "audio_class": "Heavy rain hitting glass window with distant thunder",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -20.0
    }},
    {{
      "audio_class": "Dark cinematic suspense pad with low synth drones",
      "audio_type": "MUSIC",
      "word_index": 10,
      "start_time_ms": 5000,
      "duration_ms": 10000,
      "weight_db": -14.0
    }}
  ]
}}
"""
    ),
)


# gemini_add_movie_bgms = PromptTemplate(
#     input_variables=["story_text", "speed_wps", "movie_bgms_csv","already_added_audio_cues"],
#     template=(
#         """
# You are specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.
# Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

# Story: {story_text}
# Reading Speed: {speed_wps} words per second
# Total Story Words: Count the words in the story
# Cinematic Master Model: Story Analysis & Sync Prompt
# Role & Expertise: You are a expert movie background music retriever. You need to retrieve the movie background music that best fits the story and add it to the story to make it more cinematic.

# we have already added some audio cues to the story, so you need to add movie bgm to the story and make sure to not overlap with the already added audio cues.

# Already added audio cues: {already_added_audio_cues}


# ###################### ITS NOT AT ALL NECESSARY TO HAVE MOVIE BGM IN THE STORY. IF YOU FEEL IT IS NOT NECESSARY, YOU CAN SKIP IT. ######################

# For adding movie bgm, you need to follow the following rules:
# MOVIE_BGM (Movie Background Music): Sets the mood of the scene (e.g., "Heroic soundtrack"). Duration: 2000ms–10000ms. This will be retrieved from the movie bgms data. So it might not be excatly match to description or story context.

# ### 4. MOVIE BGM RETRIEVAL
# You must identify the movie bgm that best fits the story. Use the following movie bgms data to choose the best one:
# {movie_bgms_csv}

# Identify what best sound fits to the story from above data and add it to the story. like 

# audio_class: The audio path of the movie bgm you feel is best suited from the data.
# audio_type: "MOVIE_BGM".
# word_index: The 0-based position of the word that triggers the movie bgm.
# duration_ms: The duration of the movie bgm.
# start_time_ms: The start time of the movie bgm.
# weight_db: volume adjustment (-15.0 to 6.0). 6.0 = "Defeaning/Loud".

# Try to keep the duration_ms as close to the original movie bgm duration as possible.
# Also if you choose a movie bgm from the data, you must force the audio_type to be "MOVIE_BGM" and you need to turn off the Music cue by model, ie at a time either you have movie bgm or music cue, not both.

# ### IMPORTANT: Its not at all necessary to have movie bgm in the story. If you feel it is not necessary, you can skip it.###

# You might also remove some musical elements which you don't feel necessary to be included in story after addgint the movie bgm or you can return the already added audio cues as it is.

# ### 5. OUTPUT CONSTRAINTS

# JSON ONLY: No conversational filler.

# Keep as many audio cues as you want to cover full story into audio. Focus on story context try to find audio sources, what music or sfx should be included in the story.

# Narrator Object: Include a single narrator_description at the root.

# ### EXAMPLE OF EXPECTED ANALYSIS Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

# JSON

# {{

#   "audio_cues": [
#     {{
#         "story": " The part of the story that the narrator will read with given descrpition , make sure to include pauses and breaks as per the narrator description
        
#         # you might break story into multiple parts and make seprate audio cues for each part
#         ", 
#         "narrator_description": "Tapan speaks at a moderate pace with a low-pitched, gravelly tone to convey mystery. Clear, close-sounding recording with a cold, detached emotional depth",
#         ##### audio que for narrator to know how to read the story
#         "audio_type": "NARRATOR",
#         "start_time_ms": 0,
#         "duration_ms": duration_ms,
#     }},
#     must include already Added audio cues in the story.
#     and follow below to add movie bgms
#      {{
#       "audio_class": "Audio file path of the movie bgm you feel is best suited from the data.",
#       "audio_type": "MOVIE_BGM", # FORCE TO BE MOVIE_BGM if you take movie bgm from the data
#       "start_time_ms": 10000,# best start time according to you when the movie bgm should start
#       "duration_ms": 10000,# best duration according to you when the movie bgm should end, try to keep it as close to the original movie bgm duration as possible.
#       "weight_db": 0.0 # best weight db/loudness according to you when the movie bgm should be played
#     }},

   
#   ]
# }}

# """
#     ),
# )


gemini_add_movie_bgms = PromptTemplate(
    input_variables=["story_text", "speed_wps", "bgm_candidates", "already_added_audio_cues"],
    template=(
        """
You are a Master Music Supervisor and Lead Mixing Engineer for cinematic audio.

Story: {story_text}
Reading Speed: {speed_wps} words per second

A semantic retrieval system has searched our professional movie-BGM library for tracks whose *scene vibe* best matches this story, and shortlisted the candidates below. Each candidate is a REAL track. `clip_id` is its unique identifier, `audio_prompt` describes the music itself, `retrieval_prompt` describes the scene it originally scored, `duration_sec` is its length, and `similarity` is how closely it matched (higher = closer).

Shortlisted MOVIE_BGM Candidates (already ranked by relevance):
{bgm_candidates}

Currently Existing Audio Cues:
{already_added_audio_cues}

### 1. YOUR MISSION: PICK THE BEST CANDIDATE (HYBRID)
Pick the single candidate (if any) that best fits the story's emotional arc. Do NOT pick purely by `similarity` — judge which track would actually elevate the storytelling.

* SCENARIO A (STRONG MATCH): Add ONE cue with `audio_type` "MOVIE_BGM" and set `audio_class` to the chosen candidate's **exact `clip_id`**. If you do this, you MUST DELETE any existing "audio_type": "MUSIC" cues from the existing list so the soundtracks do not clash.
* SCENARIO B (NO MATCH / SKIP): If none of the candidates fit the specific mood, DO NOT force it. Return the `already_added_audio_cues` exactly as they are without adding a MOVIE_BGM.

### 2. THE MIX HIERARCHY & AUDIO DUCKING (CRITICAL)
If you decide to add a MOVIE_BGM, calculate its `weight_db` so it does not distort or drown out the other sounds.
1. NARRATOR: The anchor, assumed to be at 0.0 dB.
2. SFX: Action sounds sit at -2.0 to -6.0 dB.
3. MOVIE_BGM: Must sit UNDER the narrator. Assign a weight of -10.0 to -15.0 dB.
4. AMBIENCE: Sits at the bottom at -18.0 to -24.0 dB.

### 3. FORMATTING THE MOVIE_BGM CUE
If adding a track, use this exact format:
- "audio_class": the chosen candidate's exact `clip_id` (NOT a description).
- "audio_type": "MOVIE_BGM"
- "start_time_ms": The logical start time based on the story's emotional shift.
- "duration_ms": Keep within the candidate's `duration_sec` when possible, unless the scene ends earlier.
- "weight_db": Calculated based on the Mix Hierarchy (e.g., -14.0).

### 4. OUTPUT CONSTRAINTS
* JSON ONLY. No markdown formatting, no conversational filler.
* Output a single JSON object with the root key "audio_cues" containing the final array of dictionaries.

### EXAMPLES OF DECISION MAKING

EXAMPLE 1: A STRONG MATCH IS FOUND
Decision: Candidate clip_id "RJAJdYw3ctw_0023" is a heroic battle theme that fits.
JSON Output:
{{
  "audio_cues": [
    {{ "audio_type": "NARRATOR", "start_time_ms": 0, "duration_ms": 5000, "narrator_description": "..." }},
    {{ "audio_type": "SFX", "audio_class": "Sword clash", "start_time_ms": 1000, "duration_ms": 1000, "weight_db": -4.0 }},
    {{
      "audio_class": "RJAJdYw3ctw_0023",
      "audio_type": "MOVIE_BGM",
      "start_time_ms": 0,
      "duration_ms": 15000,
      "weight_db": -12.0
    }}
  ]
}}
*(Note: Any original generic "MUSIC" cues were deleted and replaced by the MOVIE_BGM)*

EXAMPLE 2: NO MATCH FOUND (SKIPPING)
Decision: The story is a quiet, tense thriller, but every candidate is happy/upbeat. Skip.
JSON Output:
{{
  "audio_cues": [
    {{ "audio_type": "NARRATOR", "start_time_ms": 0, "duration_ms": 5000, "narrator_description": "..." }},
    {{ "audio_type": "AMBIENCE", "audio_class": "Quiet wind", "start_time_ms": 0, "duration_ms": 5000, "weight_db": -20.0 }}
  ]
}}
*(Note: The array is returned exactly as it was provided in the prompt, with no MOVIE_BGM added)*
"""
    ),
)


gemini_retrieve_movie_bgms = PromptTemplate(
    input_variables=["video_bytes", "mime_type","bgm_candidates"],
    template=(
        """
### 1. YOUR MISSION: PICK THE BEST CANDIDATE (HYBRID)
Pick the best suited movie background musics track from the video. Do NOT pick purely by `similarity` — judge which track would actually elevate the storytelling.

* SCENARIO A (STRONG MATCH): Add cues with `audio_type` "MOVIE_BGM" and set `audio_class` to the chosen candidate's **exact `clip_id`**.
* SCENARIO B (NO MATCH / SKIP): If none of the candidates fit the specific mood, DO NOT force it. Return the empty array.

Shortlisted MOVIE_BGM Candidates (already ranked by relevance):
{bgm_candidates}

### 2. FORMATTING THE MOVIE_BGM CUE
If adding a track, use this exact format:
- "audio_class": the chosen candidate's exact `clip_id` (NOT a description).
- "audio_type": "MOVIE_BGM"
- "start_time_ms": The logical start time based on the story's emotional shift.
- "duration_ms": Keep within the candidate's `duration_sec` when possible, unless the scene ends earlier.
- "weight_db": Calculated based on the Mix Hierarchy (e.g., -14.0).

### 4. OUTPUT CONSTRAINTS
* JSON ONLY. No markdown formatting, no conversational filler.
* Output a single JSON object with the root key "audio_cues" containing the final array of dictionaries.

### EXAMPLES OF DECISION MAKING

EXAMPLE 1: ADD MUSIC MATCH IS FOUND
Decision: Candidate clip_id "RJAJdYw3ctw_0023" is a heroic battle theme that fits.
JSON Output:
{{
  "audio_cues": [
    {{ "audio_type": "NARRATOR", "start_time_ms": 0, "duration_ms": 5000, "narrator_description": "..." }},
    {{ "audio_type": "SFX", "audio_class": "Sword clash", "start_time_ms": 1000, "duration_ms": 1000, "weight_db": -4.0 }},
    {{
      "audio_class": "RJAJdYw3ctw_0023",
      "audio_type": "MOVIE_BGM",
      "start_time_ms": 0,
      "duration_ms": 15000,
      "weight_db": -12.0
    }}
  ]
}}
*(Note: Any original generic "MUSIC" cues were deleted and replaced by the MOVIE_BGM)*


"""
    ),
)

gemini_video_audio_cues = PromptTemplate(
    input_variables=["narrator_rules"],
    template=(
        """
You are a Master Foley Artist, Cinematic Sound Designer, and Lead Mixing Engineer.
You are given a video whose original audio has been muted. Your job is to completely reconstruct, foley, and dramatize the soundscape from scratch by analyzing the visual cues frame-by-frame.

### CORE DIRECTIVES (CRITICAL)
1. **Action-Driven SFX, Not Object-Driven:** Do NOT hallucinate sounds just because an object is on screen. For example, if you see a dog sitting silently, do NOT add a barking sound. ONLY trigger a barking SFX if you visually see the dog actively opening its mouth to bark. Anchor all SFX strictly to visible kinetic impacts, physical interactions, or clear weather changes.
2. **Holistic Musical Arc:** Watch the ENTIRE video's visual progression before deciding on MUSIC cues. Understand the overarching emotional arc (e.g., does it start peaceful and turn suspenseful?). MUSIC should dictate the mood of logical narrative chapters, not jump randomly every few seconds.
3. **Dialogue / Lip-Sync:** If you see a human or character actively moving their mouth to speak, use a NARRATOR cue to dub them. Ensure the `duration_ms` of the NARRATOR cue matches the exact time window their mouth is moving.

### TIMELINE & SYNCHRONIZATION
- All times are relative to the **start of the video** (t = 0).
- `start_time_ms` and `duration_ms` are in **milliseconds**.
- At the **root** of your JSON, include `total_video_duration_ms`: your exact measurement of the video's total length in milliseconds.
- Do not trigger any cues that extend past the `total_video_duration_ms`.

### SCENE VIBE (FOR MUSIC RETRIEVAL)
At the **root** of your JSON, also include `scene_vibe`: a single rich paragraph describing the overall *emotional and cinematic* essence of the video — narrative purpose, emotional progression, cinematic mood, pacing, visual atmosphere (lighting/color), and the kind of background music that would best score it. This is NOT about literal sound effects; it captures the storytelling subtext (e.g. "a tense, slow-burn villain entrance in dim golden light, building dread toward an explosive reveal; the score should be ominous and brooding with low strings"). It is used to retrieve a real professionally-scored track that matches this vibe.

{narrator_rules}

### AUDIO TYPES
* **SFX**: Short, precise events (e.g., footsteps, glass shattering, punch, door slam, car engine revving). Duration 200–3000 ms. Must sync perfectly with visual impact.
* **AMBIENCE**: Continuous environmental bed (e.g., room tone, distant traffic, heavy rain). Start when the scene establishes; end at a hard cut or video end. 
* **MUSIC**: Generative emotional score. Usually 2000–15000 ms. 
* **MOVIE_BGM**: Do NOT use in this pass.

###  MIX HIERARCHY & LOUDNESS (weight_db)
You must mix the scene professionally. Never stack MUSIC, AMBIENCE, and heavy SFX at the same high volume.
1. **NARRATOR / DIALOGUE**: Anchor at 0.0 dB. This is the focal point when active.
2. **SFX**: -2.0 to -6.0 dB for standard actions. Only use +2.0 to +4.0 dB for massive cinematic impacts (explosions, jump scares).
3. **MUSIC**: Base level at -10.0 to -15.0 dB. *Crucial:* If a NARRATOR is speaking, duck the music further down to -20.0 dB.
4. **AMBIENCE**: Bottom of the mix to provide texture. -18.0 to -24.0 dB.

### OUTPUT FORMAT
Return **ONLY** valid JSON. Do not include markdown formatting blocks (like ```json), do not include conversational text.

{{
  "total_video_duration_ms": <integer>,
  "scene_vibe": "<One rich paragraph describing the emotional/cinematic essence of the whole video for music retrieval>",
  "audio_cues": [
    {{
      "audio_class": "<Detailed physical description for the AI audio generator, e.g., 'heavy wooden door slamming shut'>",
      "audio_type": "SFX",
      "start_time_ms": 1500,
      "duration_ms": 800,
      "weight_db": -3.5
    }},
    {{
      "story": "<Dialogue to be spoken>",
      "narrator_description": "<Voice characteristics, e.g., 'Middle-aged male, frantic, breathless, close-mic'>",
      "audio_type": "NARRATOR",
      "start_time_ms": 2500,
      "duration_ms": 3000,
      "weight_db": 0.0
    }}
  ]
}}
"""
    ),
)

gemini_video_add_movie_bgms = PromptTemplate(
    input_variables=["bgm_candidates", "already_added_audio_cues"],
    template=(
        """
You are a Master Music Supervisor and Lead Mixing Engineer for cinematic audio.

The **same video** is attached again. Use what you see together with the emotional arc of the visuals.

A semantic retrieval system has already searched our professional movie-BGM library for tracks whose *scene vibe* best matches this video, and shortlisted the candidates below. Each candidate is a REAL track from real movie scenes. `clip_id` is its unique identifier, `audio_prompt` describes the music itself, `retrieval_prompt` describes the scene it originally scored, `duration_sec` is its length, and `similarity` is how closely its scene vibe matched this video (higher = closer).

Shortlisted MOVIE_BGM Candidates (already ranked by relevance):
{bgm_candidates}

Currently Existing Audio Cues (from the prior pass — preserve unless you modify per rules below):
{already_added_audio_cues}

### 1. YOUR MISSION: PICK THE BEST CANDIDATE (HYBRID)
Pick the single candidate (if any) that best fits the video's emotional arc. Do NOT favor a track only because it has the highest `similarity` — watch the video and judge which track would actually elevate the storytelling.

* **SCENARIO A (STRONG MATCH)**: Add ONE cue with `audio_type` "MOVIE_BGM" and set `audio_class` to the chosen candidate's **exact `clip_id`**. If you add a MOVIE_BGM, you **MUST** remove any overlapping or conflicting **MUSIC** cues from the list so generated score and library BGM never clash.
* **SCENARIO B (NO MATCH)**: If none of the candidates genuinely fit the mood, DO NOT force it. Return the `already_added_audio_cues` array **unchanged**. The wrong music is worse than none.

### 2. THE MIX HIERARCHY (CRITICAL)
1. NARRATOR: anchor ~0.0 dB.
2. SFX: -2.0 to -6.0 dB.
3. MOVIE_BGM: -10.0 to -15.0 dB under narrator.
4. AMBIENCE: -18.0 to -24.0 dB.

### 3. MOVIE_BGM CUE FORMAT
- `audio_class`: the chosen candidate's exact `clip_id` (NOT a description).
- `audio_type`: "MOVIE_BGM"
- `start_time_ms`: where in the video timeline (ms) the track should begin (its emotional shift).
- `duration_ms`: how long it should play; keep it within the candidate's `duration_sec` when possible.
- `weight_db`: per the Mix Hierarchy (e.g. -12.0).

### 4. OUTPUT CONSTRAINTS
* JSON ONLY. No markdown, no filler.
* Output a single JSON object: {{ "audio_cues": [ ... final merged list ... ] }}
* Include **all** non-MUSIC cues you keep from the input; when adding MOVIE_BGM, drop conflicting MUSIC cues as required above.
"""
    ),
)

prompt_to_fill_missing_audio_cues = PromptTemplate(
    input_variables=["story_text", "audio_cues", "movie_bgms_csv"],
    template=(
        """
You are an expert audio designer and storyteller assistant, specializing in identifying and engineering cinematic soundscapes for narrative enhancement. Your task is to scrutinize the given story and its currently provided audio cues for any missing critical sound elements and meticulously engineer supplementary cues only when genuinely necessary. Your objective is not simply to fill space, but to elevate immersion and narrative clarity where audio is lacking.

Below are the resources provided to you:
* Story Text: {story_text}
* Pre-existing Audio Cues: {audio_cues}
* Reference Movie BGM Data (CSV): {movie_bgms_csv}

## 1. DETAILED GAP ANALYSIS
Thoroughly read and analyze the story in the context of the present audio cues. Methodically identify any narrative moment or sensory detail (such as dramatic actions, atmospheric transitions, or emotional cues) that is not already reflected in the provided cues. Carefully ask yourself:
- Are any important actions, impacts, or narrative turns missing a SFX cue?
- Is the ambience—such as environment, weather, or location—properly represented for every part of the story? Are there silent passages where there should be a textural background?
- Does the scene lack the proper emotional underpinning through MUSIC or, where applicable, a MOVIE_BGM from the data?
- Are crescendos, moments of suspense, or transitions (scene changes) insufficiently scored by music or background?

> Only if you uncover an audible gap, generate additional audio cues strictly for those gaps.
> If the current cues fully support the story—including all significant SFX, AMBIENCE, and MUSIC/MOVIE_BGM—do NOT create new cues. In this case, return an empty JSON array.

## 2. STRICT OUTPUT POLICY
Return **only** the newly engineered cues that are missing in the existing array. **Do not repeat or summarize any of the existing audio cues**. You must not return conversational filler or markdown—your only response is a JSON object as described below.

## 3. AUDIO CUE CONSTRUCTION & PARAMETERS

- **SFX (Sound Effects)**: Use for distinct narrative actions/impacts (e.g., footsteps, doors, objects, dramatic events). Ensure each SFX is justified by a clear textual trigger. Typical duration: 500–3000ms.
- **AMBIENCE**: Add only for persistent background environments (e.g., rain, crowd murmur, forest sounds). Duration should begin at the relevant trigger and end at the next major change or scene shift. Avoid overlapping identical types—if a new AMBIENCE begins, terminate the previous instance at its start.
- **MUSIC (Emotional Score)**: Used to indicate emotional/scenic background when MOVIE_BGM is inappropriate, unavailable, or contextually unfitting. Do not use if a valid MOVIE_BGM is present for the same time span.
- **MOVIE_BGM (Movie Background Music)**: Select and insert the single most context-appropriate BGM track from the given CSV data if the scene supports its use. The chosen track does not need to perfectly fit the narrative, but it must not distract or conflict with the story context. Its duration should match or reasonably fit the source file, and you must never play both MUSIC and MOVIE_BGM simultaneously.

All cues must specify:
- `audio_class`: Either the detailed description (“wooden door creak”, “orchestral suspense pad”) or the file path from the MOVIE_BGM data when applicable.
- `audio_type`: One of `"SFX"`, `"AMBIENCE"`, `"MUSIC"`, `"MOVIE_BGM"`.
- `word_index`: Index of the very first word in the story text that triggers this cue.
- `start_time_ms`, `duration_ms`: Exact millisecond values for cue timing and span.
- `weight_db`: Follow the strict hierarchy below.

### MIXING HIERARCHY—STRICTLY ENFORCED
- SFX (Action/Impact): -2.0 to -6.0 dB (except: huge explosions may reach +2.0 dB).
- MUSIC or MOVIE_BGM (Mood/Emotion): -10.0 to -15.0 dB. Ensure only one is present at any given moment.
- AMBIENCE: -18.0 to -24.0 dB.

## 4. MOVIE BGM SELECTION & CONFLICTS
- If using a MOVIE_BGM (selected via `audio_class` from CSV), strictly forbid any MUSIC cues at overlapping times.
- If no MOVIE_BGM contextually fits, skip it entirely.
- Movie BGM cues take precedence if their scene match is reasonable.
- When choosing MOVIE_BGM from the CSV data, copy the `audio_class` (file path) as provided and keep the duration as close to the original as practical.

## 5. NARRATOR OBJECT INTEGRATION
For every missing section that should be read aloud, include a `"narrator_description"` (defining speaker voice, delivery style, prosody, etc.) and, where helpful, suggest how to break narration into logical parts for clarity or dramatic pacing.

## 6. OUTPUT—STRICT, CLEAN JSON
- Respond ONLY with a single JSON object containing a single key `"audio_cues"` pointing to a list of new cue objects OR an empty list if nothing is missing.
- No other text, explanation, markdown, or conversational content may appear.
- All required fields must be present on every cue.
- If no cues are missing, return: `{{"audio_cues": []}}`

---

### EXAMPLE SCENARIO

_Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall."_

JSON:
{{
  "audio_cues": [
    {{
      "story": "(The part of the story to be narrated with embedded pauses and emphasis),",
      "narrator_description": "male, monotone, high-pitched, slow-paced delivery in a confined, echoey space.",
      "audio_type": "NARRATOR",
      "start_time_ms": 0,
      "duration_ms": 4000
    }},
    {{
      "audio_class": "path/to/thriller_bgm.wav",
      "audio_type": "MOVIE_BGM",
      "word_index": 8,
      "start_time_ms": 3000,
      "duration_ms": 9000,
      "weight_db": -13.0
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly with high-frequency friction",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -2.0
    }},
    {{
      "audio_class": "Intense rain hitting glass",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -19.0
    }}
  ]
}}
"""
    ),
)



# ===========================================================================
# CINEMATIC INTELLIGENCE PIPELINE
# Stage 1: video -> rich Scene Manifest
# Stage 2: Scene Manifest -> enriched Audio Cues
# ===========================================================================

MANIFEST_SYSTEM_PROMPT = (
    """You are a Cinematic Intelligence Engine: part film director, part sound designer, part music supervisor.
You watch a silent video clip and reconstruct the FULL narrative and emotional subtext behind it — not just what objects appear, but what the scene MEANS and how it should FEEL.
Crucially, you understand the difference between a triumphant hero walking into a room and a terrified victim hiding in one, even if the framing looks similar. You read lighting, framing, blocking, pacing, micro-expressions, and editing rhythm to infer emotional intent.
You NEVER hallucinate sounds from mere object presence; you reason about narrative purpose, mood, and the score a professional Hollywood director would commission for this exact moment.
You always respond with strict, valid JSON and no commentary."""
)


manifest_extraction_prompt = PromptTemplate(
    input_variables=["dialogue_script", "director_prompt"],
    template=(
        """Analyze the attached SILENT video and produce a rich Scene Manifest as strict JSON.

Optional creative inputs (use when present to deepen interpretation):
- Dialogue / script: {dialogue_script}
- Director's prompt (high-level intent): {director_prompt}

### YOUR TASK (Stage 1 of 2)
Watch the ENTIRE clip first. Then produce:
1. **video_description** — a vivid, chronological textual story of EXACTLY what happens on screen (who, where, actions, camera, lighting, pacing). Write as if narrating the clip beat-by-beat with timestamps woven in.
2. **sound_elements** — every audible layer the scene implies: ambience beds, SFX tied to visible action, dialogue windows, music chapters. Each with precise start_ms / end_ms.
3. **scene_interpretation** — emotional subtext from dialogue + director prompt + visuals: what the scene MEANS, not just what happens.
4. **scene_gravity** — why this scene exists in the story: stakes, narrative function, what would be lost if it were cut.

Measure total video length precisely in milliseconds.

Return ONLY a single JSON object with EXACTLY this shape (no markdown, no prose):
{{
  "global_metadata": {{
    "total_video_duration_ms": <integer>,
    "logline": "<one-sentence summary>",
    "genre": "<e.g. thriller, romance, action>",
    "overall_mood": "<dominant emotional tone>",
    "recommended_score_style": "<music a director would commission>"
  }},
  "video_description": "<multi-paragraph chronological account of the entire clip>",
  "scene_interpretation": {{
    "narrative_purpose": "<what this scene accomplishes>",
    "emotional_depth": "<layered feelings, subtext, character psychology>",
    "subtext": "<the unspoken truth>",
    "tension_curve": "<how tension evolves start->end>"
  }},
  "scene_gravity": {{
    "why_this_scene_matters": "<narrative reason this beat exists>",
    "emotional_weight": "<light|moderate|heavy>",
    "stakes": "<what is at risk>",
    "story_function": "<setup|payoff|turning_point|breather|climax>"
  }},
  "cinematic_pacing": {{
    "editing_rhythm": "<slow/deliberate | steady | fast/frenetic>",
    "key_beats_ms": [<integer timestamps of major visual/emotional beats>]
  }},
  "visual_palette": {{
    "lighting": "<e.g. dim golden, high-contrast noir>",
    "color_mood": "<dominant colors and what they evoke>"
  }},
  "emotional_arc": [
    {{ "start_ms": <int>, "end_ms": <int>, "label": "<mood>", "intensity": <0.0-1.0> }}
  ],
  "spatial_context": {{
    "location": "<where>",
    "ambience_bed": "<continuous environmental sound>"
  }},
  "character_dynamics": [
    {{ "who": "<role>", "is_speaking": <true|false>, "speak_windows_ms": [[<start>,<end>]] }}
  ],
  "sound_elements": [
    {{ "kind": "SFX|AMBIENCE|MUSIC|DIALOGUE", "description": "<what & why>", "start_ms": <int>, "end_ms": <int>, "importance": <0.0-1.0>, "sync_note": "<what visual action it aligns to>" }}
  ]
}}

Rules:
- All timestamps in milliseconds, relative to t=0, never exceeding total_video_duration_ms.
- List AT LEAST one AMBIENCE bed spanning most of the clip when the space implies it.
- SFX must anchor to visible kinetic action — never invent from object presence alone.
- MUSIC entries describe emotional chapters for score placement.
- DIALOGUE only when a character visibly speaks; align to mouth movement.
- Be exhaustive: a professional sound designer should be able to mix from this manifest alone.
"""
    ),
)


manifest_to_cues_prompt = PromptTemplate(
    input_variables=["manifest_json", "narrator_rules"],
    template=(
        """You are a Lead Sound Designer and Mixing Engineer. Convert the Scene Manifest below into a precise, mix-ready list of enriched Audio Cues.

Scene Manifest (authoritative source of truth):
{manifest_json}

{narrator_rules}

### CUE TYPES
- SFX: short kinetic events (200-3000ms), synced to action.
- AMBIENCE: continuous environmental bed.
- MUSIC: generative emotional score (use when no retrieved BGM fits).
- MOVIE_BGM: a real library track filled later by retrieval — emit it as a MUSIC-like cue but set "audio_type":"MUSIC" with a rich "retrieval_query"; the backend decides retrieval vs generation.
- NARRATOR: dubbed dialogue aligned to speaking windows.

### MIX HIERARCHY (weight_db)
1. NARRATOR/DIALOGUE: ~0.0 dB. 2. SFX: -2.0..-6.0 (impacts up to +4.0). 3. MUSIC/MOVIE_BGM: -10.0..-15.0 (duck to -20 under narration). 4. AMBIENCE: -18.0..-24.0.

### DUCKING PRIORITY
Assign integer "ducking_priority" (1 = most protected, ducks everything else; higher = ducks more easily). NARRATOR=1, key SFX=2, MUSIC=3, AMBIENCE=4.

### OUTPUT FORMAT
Return ONLY valid JSON (no markdown). Root object:
{{
  "total_video_duration_ms": <integer from manifest>,
  "audio_cues": [
    {{
      "cue_id": "music_001",
      "audio_class": "<generator description OR scene-music intent>",
      "audio_type": "MUSIC",
      "start_time_ms": 0,
      "duration_ms": 12000,
      "weight_db": -12.0,
      "fade_in_ms": 1500,
      "fade_out_ms": 2000,
      "ducking_priority": 3,
      "retrieval_query": "<rich scene-vibe paragraph for BGM retrieval: mood, pacing, lighting, emotional arc, desired instrumentation>",
      "mood_arc_label": "quiet_dread_to_release",
      "bpm_hint": 72,
      "key_feel": "minor",
      "instrumentation_hint": "low strings, sub drones, sparse piano",
      "energy_trajectory": "slow build then swell",
      "energy_keyframes": [{{"time_ms": 0, "db": -16.0}}, {{"time_ms": 9000, "db": -10.0}}],
      "fallback_chain": ["yt_bgm_dataset", "musicgen"]
    }},
    {{
      "cue_id": "sfx_001",
      "audio_class": "heavy wooden door slamming shut",
      "audio_type": "SFX",
      "start_time_ms": 1500,
      "duration_ms": 800,
      "weight_db": -3.5,
      "fade_in_ms": 0,
      "fade_out_ms": 120,
      "ducking_priority": 2
    }},
    {{
      "cue_id": "nar_001",
      "story": "<dialogue text>",
      "narrator_description": "<voice characteristics>",
      "audio_type": "NARRATOR",
      "start_time_ms": 2500,
      "duration_ms": 3000,
      "weight_db": 0.0,
      "ducking_priority": 1
    }}
  ]
}}

Rules:
- Only include fields that apply (SFX rarely needs retrieval_query/bpm_hint; MUSIC should always have a retrieval_query).
- Keep timings within total_video_duration_ms.
- Every MUSIC cue MUST include a detailed "retrieval_query" so the backend can search the BGM library.
"""
    ),
)


manifest_to_cues_video_prompt = PromptTemplate(
    input_variables=["manifest_json", "narrator_rules"],
    template=(
        """You are a Lead Sound Designer and Mixing Engineer. Re-watch the attached SILENT video while reading the Scene Manifest below. Convert the manifest into a precise, mix-ready list of enriched Audio Cues.

The manifest is authoritative for emotional intent; the video is authoritative for timing sync.

Scene Manifest:
{manifest_json}

{narrator_rules}

### CUE TYPES
- SFX: short kinetic events (200-8000ms), hard-synced to visible action.
- AMBIENCE: continuous environmental bed for the full scene or major sections.
- MUSIC: generative emotional score when no library track fits.
- MOVIE_BGM: emit as "audio_type":"MUSIC" with rich "retrieval_query"; backend retrieves from library.
- NARRATOR: dubbed VO aligned to speaking windows.

### MIX HIERARCHY (weight_db)
1. NARRATOR: ~0.0 dB. 2. SFX: -2.0..-6.0 (impacts up to +4.0). 3. MUSIC/MOVIE_BGM: -10.0..-15.0. 4. AMBIENCE: -18.0..-24.0.

### DUCKING PRIORITY
NARRATOR=1, key SFX=2, MUSIC=3, AMBIENCE=4.

### OUTPUT FORMAT
Return ONLY valid JSON (no markdown):
{{
  "total_video_duration_ms": <integer>,
  "audio_cues": [
    {{
      "cue_id": "amb_001",
      "audio_class": "narrow urban alleyway ambience with distant traffic",
      "audio_type": "AMBIENCE",
      "start_time_ms": 0,
      "duration_ms": 25500,
      "weight_db": -18.0,
      "fade_in_ms": 2000,
      "fade_out_ms": 2000,
      "ducking_priority": 4
    }},
    {{
      "cue_id": "music_001",
      "audio_class": "cinematic action score intent",
      "audio_type": "MUSIC",
      "start_time_ms": 0,
      "duration_ms": 25500,
      "weight_db": -12.0,
      "fade_in_ms": 1500,
      "fade_out_ms": 3000,
      "ducking_priority": 3,
      "retrieval_query": "<rich paragraph: mood, pacing, lighting, emotional arc, instrumentation>",
      "mood_arc_label": "anticipation_to_climax",
      "bpm_hint": 120,
      "key_feel": "minor",
      "instrumentation_hint": "strings, percussion, synth pulses",
      "energy_trajectory": "build then peak at explosion",
      "energy_keyframes": [{{"time_ms": 0, "db": -16}}, {{"time_ms": 15000, "db": -8}}],
      "fallback_chain": ["yt_bgm_dataset", "musicgen"]
    }},
    {{
      "cue_id": "sfx_001",
      "audio_class": "heavy automatic gunfire bursts",
      "audio_type": "SFX",
      "start_time_ms": 6000,
      "duration_ms": 8000,
      "weight_db": -3.0,
      "fade_in_ms": 50,
      "fade_out_ms": 500,
      "ducking_priority": 2
    }},
    {{
      "cue_id": "nar_001",
      "story": "<dialogue>",
      "narrator_description": "<voice>",
      "audio_type": "NARRATOR",
      "start_time_ms": 500,
      "duration_ms": 4500,
      "weight_db": 0.0,
      "ducking_priority": 1
    }}
  ]
}}

Rules:
- Include cues for EVERY important sound_element in the manifest.
- Always include a full-span AMBIENCE bed unless the scene is truly silent space.
- Include ONE primary MUSIC/MOVIE_BGM cue spanning the emotional arc with detailed retrieval_query.
- Sync SFX start_time_ms to the exact visual action in the video.
- Keep all timings within total_video_duration_ms.
- Use descriptive audio_class strings suitable for generative models.
"""
    ),
)


alignment_prediction_prompt = PromptTemplate(
    input_variables=["story_prompt", "audio_classes", "whisper_json"],
    template=(
        """
        You are a helpful assistant that predicts the alignment of the audio classes with the story prompt.
        I have a Story text as : {story_prompt}
        I have generated Audio Cues: {audio_classes}
        I have generated Whisper JSON: {whisper_json}
        
        You need to predict the alignment of the audio classes with the story prompt.
        You need to return the alignment in the following format:
        
          "audio_cues": [
          {{
            "id": <same id as the audio class>,
            "audio_class": <same as the audio class>,
            "audio_type": <same as the audio type>,
            "start_time_ms": 5000, # starting time in milliseconds
            "duration_ms": 10000, # duration in milliseconds
            "weight_db": 0.0,
            "fade_ms": 500
          }}, 
          ]
          
          example
            "story_prompt": "A helmet-clad soldier cautiously navigates a grimy, dimly lit urban corridor before being brutally ambushed by a bloodied operative, who then, protecting a young boy, plunges into a chaotic close-quarters gunfight against multiple assailants."
            "whisper.json":
            [{{"word": "A", "start": 0.0, "end": 0.18}}, {{"word": "helmet,", "start": 0.18, "end": 1.42}}, {{"word": "clad", "start": 1.42, "end": 1.9}}, {{"word": "soldier", "start": 1.9, "end": 2.34}}, {{"word": "cautiously", "start": 2.34, "end": 3.02}}, {{"word": "navigates", "start": 3.02, "end": 3.6}}, {{"word": "a", "start": 3.6, "end": 3.72}}, {{"word": "grimy,", "start": 3.72, "end": 4.34}}, {{"word": "dimly", "start": 4.34, "end": 4.62}}, {{"word": "lit", "start": 4.62, "end": 4.86}}, {{"word": "urban", "start": 4.86, "end": 5.22}}, {{"word": "corridor", "start": 5.22, "end": 5.62}}, {{"word": "before", "start": 5.62, "end": 6.22}}, {{"word": "being", "start": 6.22, "end": 6.54}}, {{"word": "brutally", "start": 6.54, "end": 6.92}}, {{"word": "ambushed", "start": 6.92, "end": 7.6}}, {{"word": "by", "start": 7.6, "end": 7.7}}, {{"word": "a", "start": 7.7, "end": 7.86}}, {{"word": "bloodied", "start": 7.86, "end": 8.14}}, {{"word": "operative,", "start": 8.14, "end": 9.42}}, {{"word": "who", "start": 9.42, "end": 9.48}}, {{"word": "then", "start": 9.48, "end": 9.7}}, {{"word": "protecting", "start": 9.7, "end": 10.22}}, {{"word": "a", "start": 10.22, "end": 10.44}}, {{"word": "young", "start": 10.44, "end": 10.64}}, {{"word": "boy", "start": 10.64, "end": 10.98}}, {{"word": "plunges", "start": 10.98, "end": 11.96}}, {{"word": "into", "start": 11.96, "end": 12.2}}, {{"word": "a", "start": 12.2, "end": 12.38}}, {{"word": "chaotic", "start": 12.38, "end": 12.7}}, {{"word": "close", "start": 12.7, "end": 13.1}}, {{"word": "-quarters", "start": 13.1, "end": 13.48}}, {{"word": "gunfight", "start": 13.48, "end": 14.04}}, {{"word": "against", "start": 14.04, "end": 14.56}}, {{"word": "multiple", "start": 14.56, "end": 15.1}}, {{"word": "assailants.", "start": 15.1, "end": 16.0}}]
            "audio_cues": [
              {{ "id": 1, "audio_class": "Distant urban street ambience", "audio_type": "AMBIENCE", "starting_time": 0.0, "duration": 16.0, "weight_db": -35.0 }},
              {{ "id": 2, "audio_class": "Tense synth drone with subtle rhythmic percussion", "audio_type": "MUSIC", "starting_time": 0.0, "duration": 8.8, "weight_db": -25.0 }},
              {{ "id": 3, "audio_class": "Heavy tactical footsteps and gear rustle", "audio_type": "AMBIENCE", "starting_time": 0.0, "duration": 8.5, "weight_db": -20.0 }},
              {{ "id": 4, "audio_class": "Brutal melee combat impacts and vocal grunts", "audio_type": "SFX", "starting_time": 8.7, "duration": 6.5, "weight_db": -10.0 }},
              {{ "id": 5, "audio_class": "Pistol slide rack and reload click", "audio_type": "SFX", "starting_time": 20.0, "duration": 0.5, "weight_db": -12.0 }},
              {{ "id": 6, "audio_class": "Deep male voice (low dialogue, 'Come on')", "audio_type": "NARRATOR", "starting_time": 20.5, "duration": 0.5, "weight_db": -18.0 }},
              {{ "id": 7, "audio_class": "Rapid gunfire, body impacts, and close-quarters combat SFX", "audio_type": "SFX", "starting_time": 25.9, "duration": 4.1, "weight_db": -7.0 }},
              {{ "id": 8, "audio_class": "Aggressive percussive action music swell", "audio_type": "MUSIC", "starting_time": 25.5, "duration": 4.5, "weight_db": -10.0 }}
            ],
            the output should be something like this:
            "audio_cues": [
              {{ "id": 1, "audio_class": "Distant urban street ambience", "audio_type": "AMBIENCE", "start_time_ms": 0, "duration_ms": 8000, "weight_db": -35.0, "fade_ms": 500 }},
              {{ "id": 2, "audio_class": "Tense synth drone with subtle rhythmic percussion", "audio_type": "MUSIC", "start_time_ms": 0, "duration_ms": 4690, "weight_db": -25.0, "fade_ms": 500 }},
              {{ "id": 3, "audio_class": "Heavy tactical footsteps and gear rustle", "audio_type": "AMBIENCE", "start_time_ms": 0, "duration_ms": 4250, "weight_db": -20.0, "fade_ms": 500 }},
              {{ "id": 4, "audio_class": "Brutal melee combat impacts and vocal grunts", "audio_type": "SFX", "start_time_ms": 4690, "duration_ms": 1810, "weight_db": -10.0, "fade_ms": 500 }},
              {{ "id": 5, "audio_class": "Pistol slide rack and reload click", "audio_type": "SFX", "start_time_ms": 10600, "duration_ms": 500, "weight_db": -12.0, "fade_ms": 500 }},
              {{ "id": 6, "audio_class": "Deep male voice (low dialogue, 'Come on')", "audio_type": "NARRATOR", "start_time_ms": 11000, "duration_ms": 500, "weight_db": -18.0, "fade_ms": 500 }},
              {{ "id": 7, "audio_class": "Rapid gunfire, body impacts, and close-quarters combat SFX", "audio_type": "SFX", "start_time_ms": 13810, "duration_ms": 2190, "weight_db": -7.0, "fade_ms": 500 }},
              {{ "id": 8, "audio_class": "Aggressive percussive action music swell", "audio_type": "MUSIC", "start_time_ms": 15000, "duration_ms": 1000, "weight_db": -10.0, "fade_ms": 500 }}
            ],
        """
    ),
)


gemini_pick_bgm_from_candidates = PromptTemplate(
    input_variables=["query_text", "bgm_candidates"],
    template=(
        """Scene/query: {query_text}

Candidate BGMs (pick exactly one by clip_id, or null if none fit):
{bgm_candidates}

Return JSON only: {{"clip_id": "<id>"}} or {{"clip_id": null}}"""
    ),
)




gemini_explain_textual_story_prompt = PromptTemplate(
    input_variables=["textual_story"],
    template=(
        """
You are a senior story analyst and cinematic sound supervisor.
Your job is to deeply unpack a textual story so a sound designer can score it.

Story:
{textual_story}

Analyze the story carefully and return THREE fields:

1) story_gravity
Explain WHY this passage matters in narrative terms. Be specific. Prefer one clear role, e.g.:
- setup / world-building
- character establishment
- conflict ignition
- rising tension
- turning point / reveal
- emotional breather
- climax
- payoff / resolution
Say what would be lost if this passage were removed.

2) story_interpretation
Write a cinematic reading of the scene: mood, subtext, visual atmosphere, pacing, and emotional arc.
Describe it as if briefing a film director — vivid, concrete, impactful. 3–6 sentences.

3) audio_asthetics
Describe the desired sonic world for scoring this story: instrumentation palette, energy curve,
texture (sparse vs dense), harmonic color (major/minor/modal), and mix priorities
(what should feel intimate vs epic). This will drive BGM retrieval. 3–6 sentences.

Return JSON ONLY (no markdown, no commentary):
{{
  "story_gravity": "<narrative role and why it matters>",
  "story_interpretation": "<cinematic interpretation>",
  "audio_asthetics": "<desired sonic / scoring aesthetics>"
}}
"""
    ),
)


gemini_shortlist_movie_bgms_prompt = PromptTemplate(
    input_variables=["textual_story", "bgm_candidates"],
    template=(
        """
You are a music supervisor shortlisting movie BGM tracks for a textual story.

Story context (JSON):
{textual_story}

Candidate tracks (JSON). Each candidate includes:
- clip_id: unique id (MUST be preserved exactly)
- bgm_path: dataset path
- bgm_description / audio_prompt: what the MUSIC sounds like (instruments, mood, energy) — PRIMARY filter
- scene_description / retrieval_prompt: scene vibe the track was annotated for
- duration_sec / duration_ms: native length of the original BGM file (from dataset metadata)
- similarity: retrieval score

Candidates:
{bgm_candidates}

Selection rules:
1. Read bgm_description carefully. DROP any track whose music character does not fit the story mood
   (e.g. reject heroic/action anthems for a quiet romantic rain scene; reject comedy cues for tragedy).
2. Also check scene_description against story_gravity / story_interpretation / audio_asthetics.
3. Prefer tracks whose native duration_ms can cover a meaningful story beat without extreme stretch/trim.
4. Prefer diversity of mood if multiple beats need coverage; drop near-duplicates.
5. You may return fewer tracks than given — returning zero is OK if nothing fits. Never invent clip_ids.
6. In the output, echo bgm_description, duration_sec, and duration_ms from the chosen candidates unchanged.

Return JSON ONLY:
{{
  "bgm_candidates": [
    {{
      "clip_id": "<exact clip_id>",
      "bgm_path": "<bgm_path>",
      "bgm_description": "<music description>",
      "duration_sec": <number>,
      "duration_ms": <integer>
    }}
  ]
}}
"""
    ),
)


gemma_create_textual_story_prompt = PromptTemplate(
    input_variables=["theme", "narrators_json", "audio_asthetics", "notes"],
    template=(
        """
You are a screenwriter and cinematic sound supervisor.
The user does NOT have a finished story. Invent one from their brief, then explain it for scoring.

THEME (required):
{theme}

NARRATORS (JSON array). Each item may include:
- name (optional label)
- narrator_description (REQUIRED voice / delivery style)
- dialogues (OPTIONAL exact lines; weave these in when present; invent lines when missing)
{narrators_json}

USER AUDIO AESTHETICS (optional — honour if non-empty, else invent):
{audio_asthetics}

EXTRA NOTES (optional creative intent):
{notes}

RULES:
1. Write a short cinematic STORY (roughly 80-250 words) that fits the theme.
2. If multiple narrators are listed, the story MUST feature multiple speakers / dialogue turns.
3. Use every provided dialogue line (paraphrase lightly only if needed for flow). Invent any missing spoken lines that fit each narrator_description.
4. Make speakers clear in the prose (attribution, quotes, or distinct voice beats) so a sound designer can split NARRATOR cues per speaker.
5. director_prompt MUST name each narrator (by name or role), summarize their voice from narrator_description, and state that multiple narrators speak when applicable.
6. Also produce story_gravity, story_interpretation, and audio_asthetics (same role as an explain-story pass).

Return JSON ONLY (no markdown):
{{
  "story": "<full story text with clear multi-speaker dialogue when applicable>",
  "director_prompt": "<director brief that preserves narrator cast and voices>",
  "story_gravity": "<narrative role and why it matters>",
  "story_interpretation": "<cinematic interpretation, 3–6 sentences>",
  "audio_asthetics": "<desired sonic / scoring aesthetics, 3–6 sentences>"
}}
"""
    ),
)


gemini_decide_audio_cues_prompt_without_narrator = PromptTemplate(
    input_variables=["textual_story", "bgm_candidates"],
    template=(
        """
You are a Master Sound Designer and Lead Mixing Engineer.
Extract a precise, mix-ready audio cue list for a textual story.

IMPORTANT: Do NOT include any NARRATOR cues. Narration is disabled for this request.
Score with SFX, AMBIENCE, MUSIC, and MOVIE_BGM only.

═══════════════════════════════════════
STORY CONTEXT
═══════════════════════════════════════
{textual_story}

═══════════════════════════════════════
MOVIE BGM CANDIDATES
═══════════════════════════════════════
Each candidate includes:
- clip_id, duration_sec, duration_ms, similarity
- bgm_description / audio_prompt: what the music SOUNDS like (use this to accept/reject)
- scene_description / retrieval_prompt: original scene vibe

FILTER RULE (CRITICAL): Before placing any MOVIE_BGM, read bgm_description.
Include the track ONLY if its musical character fits the story mood, gravity, interpretation,
and audio_asthetics. Reject mismatches (e.g. triumphant brass for intimate melancholy;
horror drones for a warm fairy-tale romance). Prefer generative MUSIC over a bad MOVIE_BGM.
You may omit ALL candidates if none fit.
{bgm_candidates}

═══════════════════════════════════════
1. AUDIO CUE TYPES
═══════════════════════════════════════
* SFX — short action punctuation (door creak, footstep, glass). Typical duration_ms: 500–3000.
  audio_class = vivid text description for a generative SFX model.
* AMBIENCE — continuous environment bed (rain, room tone, wind). Span from trigger until scene change / next ambience.
  audio_class = vivid environment description.
* MUSIC — generative emotional score (NOT from the BGM library). Typical duration_ms: 2000–12000.
  audio_class = vivid music description (instruments, mood, energy).
* MOVIE_BGM — retrieved library track. audio_type MUST be "MOVIE_BGM".
  audio_class MUST be the exact candidate clip_id (never a path, never a prose description, never MUSIC).

DURATION CAP: the generative model cannot render more than 20000 ms in one pass.
Prefer duration_ms <= 20000 for SFX, AMBIENCE, and MOVIE_BGM.
MUSIC may be longer than 20000 ms, and any cue may run past 20000 ms when a cut would break continuity.

═══════════════════════════════════════
2. MOVIE_BGM DURATION RULE (CRITICAL)
═══════════════════════════════════════
Each candidate includes its native length as duration_ms (from dataset metadata).
When you place a MOVIE_BGM cue:
* Set cue duration_ms CLOSE to that candidate's duration_ms (prefer within ±10–15%).
* Do NOT invent arbitrary lengths like always 10000ms.
* Prefer placing the cue on a story beat whose needed coverage roughly matches the track length.
* Slight trim is OK if the story beat is shorter; avoid extreme stretch beyond ~+15% of native duration_ms.
* If no candidate fits the needed beat length and mood, prefer MUSIC (generative) instead of forcing a bad MOVIE_BGM.

═══════════════════════════════════════
3. TIMING & SYNC
═══════════════════════════════════════
* word_index: 0-based word that triggers the sound (optional if start_time_ms is given).
* start_time_ms: exact start on the reading timeline.
* duration_ms: exact length. Same-type overlapping AMBIENCE/MUSIC should cut off the previous one.
* Cover the story with intentional sparsity — every cue must earn its place.

═══════════════════════════════════════
4. MIX HIERARCHY & DUCKING
═══════════════════════════════════════
Evaluate overlapping sounds at the same time. Assign weight_db:
1. SFX: -2.0 to -6.0 dB (jump scares/explosions may peak +2.0 to +4.0)
2. MUSIC / MOVIE_BGM: -10.0 to -15.0 dB
3. AMBIENCE: -18.0 to -24.0 dB floor

Never stack Ambience + Music + SFX all above -5.0 dB together.

═══════════════════════════════════════
5. OUTPUT CONSTRAINTS
═══════════════════════════════════════
* JSON ONLY. No markdown fences, no commentary.
* NEVER output audio_type "NARRATOR".
* For MOVIE_BGM: include only clip_ids from the candidate list; set duration_ms near the candidate's duration_ms.
* Prefer a lean, cinematic cue list over noisy over-scoring.

═══════════════════════════════════════
EXAMPLE
═══════════════════════════════════════
Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall."
Candidate includes clip_id "1RMEGiTKokk_0023" with duration_ms 9640.

JSON
{{
  "audio_cues": [
    {{
      "audio_class": "Heavy wooden door creaking open slowly",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -4.0
    }},
    {{
      "audio_class": "Heavy rain hitting glass window with distant thunder",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -20.0
    }},
    {{
      "audio_class": "Dark cinematic suspense pad with low synth drones",
      "audio_type": "MUSIC",
      "word_index": 10,
      "start_time_ms": 5000,
      "duration_ms": 10000,
      "weight_db": -14.0
    }},
    {{
      "audio_class": "1RMEGiTKokk_0023",
      "audio_type": "MOVIE_BGM",
      "start_time_ms": 2000,
      "duration_ms": 9640,
      "weight_db": -12.0
    }}
  ]
}}
"""
    ),
)


gemini_decide_audio_cues_prompt_with_narrator = PromptTemplate(
    input_variables=["textual_story", "bgm_candidates"],
    template=(
        """
You are a Master Sound Designer, Narrative Director, and Lead Mixing Engineer.
Extract a precise, mix-ready audio cue list for a textual story WITH narration enabled.

You MUST include one or more NARRATOR cues that read the story (or natural story segments) aloud.
Each NARRATOR cue needs: story, narrator_description, audio_type="NARRATOR", start_time_ms, duration_ms.
Align SFX / AMBIENCE / MUSIC / MOVIE_BGM under the spoken timeline.

═══════════════════════════════════════
STORY CONTEXT
═══════════════════════════════════════
{textual_story}

═══════════════════════════════════════
MOVIE BGM CANDIDATES
═══════════════════════════════════════
Each candidate includes:
- clip_id, duration_sec, duration_ms, similarity
- bgm_description / audio_prompt: what the music SOUNDS like (use this to accept/reject)
- scene_description / retrieval_prompt: original scene vibe

FILTER RULE (CRITICAL): Before placing any MOVIE_BGM, read bgm_description.
Include the track ONLY if its musical character fits the story mood, gravity, interpretation,
and audio_asthetics. Reject mismatches (e.g. triumphant brass for intimate melancholy;
horror drones for a warm fairy-tale romance). Prefer generative MUSIC over a bad MOVIE_BGM.
You may omit ALL candidates if none fit. You can add multiple MOVIE_BGM cues if you think it fits the story and mood.
{bgm_candidates}

═══════════════════════════════════════
1. AUDIO CUE TYPES
═══════════════════════════════════════
* NARRATOR — spoken voice-over. Required when narration is enabled.
  Fields: story (text to speak; may include pause punctuation), narrator_description
  (voice age/gender, pace, pitch, emotion, mic distance), start_time_ms, duration_ms.
  weight_db defaults to 0.0 (mix anchor). Prefer multiple NARRATOR cues when speakers differ.
* SFX — short action punctuation. Typical duration_ms: 500–3000. audio_class = vivid description. create many high quality sfx cues for the story. try to cover all the potential sound in stroy
* AMBIENCE — continuous environment bed. audio_class = vivid environment description.
* MUSIC — generative emotional score. Typical duration_ms: 2000–12000. audio_class = music description.
* MOVIE_BGM — retrieved library track. audio_type MUST be "MOVIE_BGM".
  audio_class MUST be the exact candidate clip_id (never a path, never prose, never MUSIC).

DURATION CAP: the generative model cannot render more than 20000 ms in one pass.
Prefer duration_ms <= 20000 for SFX, AMBIENCE, and MOVIE_BGM. Narration is not under this cap.
MUSIC may be longer than 20000 ms, and any cue may run past 20000 ms when a cut would break continuity.

create many high quality sfx cues for the story. try to cover all the potential sound in stroy

═══════════════════════════════════════
2. MOVIE_BGM DURATION RULE (CRITICAL)
═══════════════════════════════════════
Each candidate includes its native length as duration_ms (from dataset metadata).
When you place a MOVIE_BGM cue:
* Set cue duration_ms CLOSE to that candidate's duration_ms (prefer within ±10–15%).
* Do NOT invent arbitrary lengths like always 10000ms.
* Prefer placing the cue on a story beat whose needed coverage roughly matches the track length.
* Slight trim is OK if the story beat is shorter; avoid extreme stretch beyond ~+15% of native duration_ms.
* If no candidate fits, prefer MUSIC (generative) instead of forcing a bad MOVIE_BGM.

═══════════════════════════════════════
3. TIMING & SYNC
═══════════════════════════════════════
* word_index: 0-based word that triggers the sound (optional if start_time_ms is given).
* start_time_ms: exact start on the reading / narration timeline.
* duration_ms: exact length. Same-type overlapping AMBIENCE/MUSIC should cut off the previous one.
* Sync SFX to narrated actions; keep ambience under the full spoken span when appropriate.

═══════════════════════════════════════
4. MIX HIERARCHY & DUCKING
═══════════════════════════════════════
Evaluate overlapping sounds at the same time. Assign weight_db:
1. NARRATOR: ~0.0 dB anchor (clearest element)
2. SFX: -2.0 to -6.0 dB (jump scares/explosions may peak +2.0 to +4.0)
3. MUSIC / MOVIE_BGM: -10.0 to -15.0 dB under voice
4. AMBIENCE: -18.0 to -24.0 dB floor

Duck MUSIC/AMBIENCE under speech; allow slight swell in long pauses.
Never stack Ambience + Music + SFX all above -5.0 dB together.

═══════════════════════════════════════
5. OUTPUT CONSTRAINTS
═══════════════════════════════════════
* JSON ONLY. No markdown fences, no commentary.
* Include at least one NARRATOR cue covering the story text.
* For MOVIE_BGM: only clip_ids from candidates; duration_ms near candidate duration_ms.
* Prefer a lean, cinematic cue list.

MULTI-NARRATOR RULE (CRITICAL):
- If the story contains dialogue / multiple speakers, OR director_prompt names multiple narrators /
  voices, emit ONE NARRATOR cue PER speaker segment (not one cue for the whole story).
- Each cue's `story` is ONLY that speaker's lines (or that segment of VO).
- Each cue's `narrator_description` must match that speaker's voice (from director_prompt or
  inferred character voice). Different speakers MUST get different descriptions.
- Give non-overlapping start_time_ms / duration_ms along the reading timeline so cues play in
  speaking order. Do NOT stretch every narrator cue to the full story duration.
- A single omniscient VO with no dialogue may still be one NARRATOR cue.


═══════════════════════════════════════
EXAMPLE
═══════════════════════════════════════
Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall."
Candidate includes clip_id "1RMEGiTKokk_0023" with duration_ms 9640.

JSON
{{
  "audio_cues": [
    {{
      "story": "The door creaked open. Rain lashed against the window as he stepped into the cold hall.",
      "narrator_description": "Warm mid adult male, moderate pace, soft intimate close-mic, quiet room.",
      "audio_type": "NARRATOR",
      "start_time_ms": 0,
      "duration_ms": 7500,
      "weight_db": 0.0
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -4.0
    }},
    {{
      "audio_class": "Heavy rain hitting glass window with distant thunder",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -20.0
    }},
    {{
      "audio_class": "1RMEGiTKokk_0023",
      "audio_type": "MOVIE_BGM",
      "start_time_ms": 2000,
      "duration_ms": 9640,
      "weight_db": -12.0
    }}
  ]
}}
"""
    ),
)







# Visual Story Prompts
# Attached video is ground truth. Text fields are compact editing/retrieval aids.

gemini_explain_visual_story_prompt = PromptTemplate(
    input_variables=["director_prompt", "video_duration_ms"],
    template=(
        """
You are a senior cinematic sound supervisor watching the ATTACHED VIDEO.
The video is the ground truth. Produce a compact, editable analysis for scoring.

Client duration hint (ms, may be approximate): {video_duration_ms}
Director's creative intent (optional): {director_prompt}

Watch carefully. Note visible actions, weapons, weather, vehicles, crowd energy,
and any continuous vs one-shot events. Be concrete (e.g. "belt-fed machine gun"
not "gunfire"; "torrential rain on metal roof" not "rain").

Return FOUR fields as JSON ONLY (no markdown):

1) video_content
A timestamped beat sheet covering the FULL clip. Use millisecond ranges, e.g.:
[0–2400ms] ...
[2400–6100ms] ...
For each beat: who/what is on screen, key actions, continuous sounds that should
exist (sustained fire, engines, rain), and mood shifts. Keep under ~2500 chars.

2) scene_gravity
WHY this clip matters narratively. Prefer one clear role, e.g.:
setup / world-building, character establishment, conflict ignition, rising tension,
turning point / reveal, emotional breather, climax, payoff / resolution.
Say what would be lost if this clip were silent or removed.

3) scene_interpretation
Cinematic reading: mood, subtext, visual atmosphere, pacing, emotional arc.
Brief a film director — vivid, concrete. 3–6 sentences.

4) audio_asthetics
Desired sonic world: instrumentation, energy curve, texture (sparse vs dense),
harmonic color, mix priorities (intimate vs epic). Drives BGM retrieval. 3–6 sentences.

JSON schema:
{{
  "video_content": "<timestamped beat sheet>",
  "scene_gravity": "<narrative role and why it matters>",
  "scene_interpretation": "<cinematic interpretation>",
  "audio_asthetics": "<desired sonic / scoring aesthetics>"
}}
"""
    ),
)


gemini_shortlist_movie_bgms_visual_prompt = PromptTemplate(
    input_variables=["visual_story", "bgm_candidates"],
    template=(
        """
You are a music supervisor shortlisting movie BGM tracks for a VIDEO clip.
The ATTACHED VIDEO is ground truth for mood, pacing, and energy. The JSON below
is a compact summary the user may have edited — use it as secondary context.

Visual story context (JSON, compact):
{visual_story}

Candidate tracks (JSON). Each candidate includes:
- clip_id: unique id (MUST be preserved exactly)
- bgm_path: dataset path
- bgm_description / audio_prompt: what the MUSIC sounds like — PRIMARY filter
- scene_description / retrieval_prompt: scene vibe the track was annotated for
- duration_sec / duration_ms: native length of the original BGM file
- similarity: retrieval score

Candidates:
{bgm_candidates}

Selection rules:
1. Watch the video. DROP tracks whose musical character clashes with what you SEE
   (e.g. reject comedy cues for a grim firefight; reject horror drones for warm romance).
2. Cross-check bgm_description against scene_gravity / scene_interpretation / audio_asthetics.
3. Prefer tracks whose native duration_ms can cover a meaningful on-screen beat
   without extreme stretch/trim.
4. Prefer diversity if multiple beats need coverage; drop near-duplicates.
5. Returning fewer tracks — or zero — is OK. Never invent clip_ids.
6. Echo bgm_description, duration_sec, and duration_ms from chosen candidates unchanged.

Return JSON ONLY:
{{
  "bgm_candidates": [
    {{
      "clip_id": "<exact clip_id>",
      "bgm_path": "<bgm_path>",
      "bgm_description": "<music description>",
      "duration_sec": <number>,
      "duration_ms": <integer>
    }}
  ]
}}
"""
    ),
)


gemini_decide_audio_cues_prompt_without_narrator_video = PromptTemplate(
    input_variables=["visual_story", "bgm_candidates", "video_duration_ms"],
    template=(
        """
You are a Master Sound Designer and Lead Mixing Engineer scoring the ATTACHED VIDEO.
The video is ground truth for sync. The JSON below is a compact editable summary —
do NOT invent events that are not visible or implied on screen.

IMPORTANT: Do NOT include any NARRATOR cues. Score with SFX, AMBIENCE, MUSIC, MOVIE_BGM only.
Total video duration (ms): {video_duration_ms}
Clamp every cue so start_time_ms + duration_ms <= total video duration.

═══════════════════════════════════════
STORY CONTEXT (compact; video is primary)
═══════════════════════════════════════
{visual_story}

═══════════════════════════════════════
MOVIE BGM CANDIDATES
═══════════════════════════════════════
Each candidate includes:
- clip_id, duration_sec, duration_ms, similarity
- bgm_description / audio_prompt: what the music SOUNDS like (use to accept/reject)
- scene_description / retrieval_prompt: original scene vibe

FILTER RULE: Include MOVIE_BGM ONLY if musical character fits the video mood,
scene_gravity, scene_interpretation, and audio_asthetics. Prefer generative MUSIC
over a bad MOVIE_BGM. You may omit ALL candidates if none fit.
{bgm_candidates}

═══════════════════════════════════════
AUDIO_CLASS PRECISION (CRITICAL FOR GENERATIVE SFX)
═══════════════════════════════════════
audio_class is fed to a generative audio model. Vague prompts produce wrong sounds
(e.g. "gunfire" → pistol clicks instead of sustained machine-gun fire).

For ANY continuous / looping / sustained sound (machine-gun fire, rain, engines,
alarms, crowd roar, helicopter blades):
* Prefix audio_class with "CONTINUOUS: "
* Name the exact source (weapon model class, weather type, vehicle)
* Specify rate / mechanism (e.g. "belt-fed automatic, ~750 RPM, sustained bursts")
* Specify spatial perspective (close / mid / distant) and environment interaction
* Explicitly NEGATE wrong characters (e.g. "NOT single-shot pistol, NOT semi-auto")
* Set duration_ms to cover the on-screen span. Keep it at or under 20000 ms unless it is MUSIC or continuity needs one unbroken bed.

For one-shot SFX: still be specific (material, size, force, space).

═══════════════════════════════════════
1. AUDIO CUE TYPES
═══════════════════════════════════════
* SFX — action punctuation OR continuous action beds when needed.
  Continuous SFX (sustained fire, etc.) may span thousands of ms; one-shots 500–3000.
* AMBIENCE — continuous environment bed. Span until scene change / next ambience.
* MUSIC — generative emotional score (NOT from the BGM library).
* MOVIE_BGM — library track. audio_type MUST be "MOVIE_BGM".
  audio_class MUST be the exact candidate clip_id (never a path or prose).

═══════════════════════════════════════
2. MOVIE_BGM DURATION RULE
═══════════════════════════════════════
Set cue duration_ms close to the candidate's native duration_ms (±10–15%).
Do NOT invent arbitrary lengths like always 10000ms.
If no candidate fits beat length and mood, prefer MUSIC instead.

═══════════════════════════════════════
3. TIMING & SYNC (VIDEO TIMELINE)
═══════════════════════════════════════
* start_time_ms: exact start on the VIDEO timeline (milliseconds from t=0).
* duration_ms: exact length. Align to visible events you see in the video.
* Same-type overlapping AMBIENCE/MUSIC should cut off the previous one.
* Every cue must earn its place — lean cinematic scoring, not noisy over-scoring.
* Do NOT use word_index (this is video, not text reading).

═══════════════════════════════════════
DURATION CAP (20 SECONDS)
═══════════════════════════════════════
The generative model cannot render more than 20000 ms in one pass.
Prefer duration_ms <= 20000 for SFX, AMBIENCE, and MOVIE_BGM.
MUSIC may be longer than 20000 ms.
Any cue may also run past 20000 ms when a shorter cut would break continuity
(a bed that must keep playing). Otherwise keep each cue at or under 20000 ms.

═══════════════════════════════════════
4. MIX HIERARCHY & DUCKING
═══════════════════════════════════════
weight_db:
1. SFX: -2.0 to -6.0 dB (explosions may peak +2.0 to +4.0)
2. MUSIC / MOVIE_BGM: -10.0 to -15.0 dB
3. AMBIENCE: -18.0 to -24.0 dB floor
Never stack Ambience + Music + SFX all above -5.0 dB together.

═══════════════════════════════════════
5. OUTPUT CONSTRAINTS
═══════════════════════════════════════
* JSON ONLY. No markdown fences, no commentary.
* NEVER output audio_type "NARRATOR".
* Include total_video_duration_ms matching the video length.
* For MOVIE_BGM: only clip_ids from the candidate list.

EXAMPLE (continuous machine-gun + rain):
{{
  "total_video_duration_ms": 12000,
  "audio_cues": [
    {{
      "audio_class": "CONTINUOUS: belt-fed light machine gun, sustained automatic fire ~750 RPM, close mid-field, brass casings, muzzle blast, NOT single-shot pistol, NOT semi-auto handgun",
      "audio_type": "SFX",
      "start_time_ms": 800,
      "duration_ms": 6500,
      "weight_db": -4.0
    }},
    {{
      "audio_class": "CONTINUOUS: heavy rain on concrete and metal, dense mid-close ambience, distant thunder rolls",
      "audio_type": "AMBIENCE",
      "start_time_ms": 0,
      "duration_ms": 12000,
      "weight_db": -20.0
    }},
    {{
      "audio_class": "Dark tense cinematic percussion and low brass pulses, rising urgency",
      "audio_type": "MUSIC",
      "start_time_ms": 500,
      "duration_ms": 10000,
      "weight_db": -14.0
    }},
    {{
      "audio_class": "1RMEGiTKokk_0023",
      "audio_type": "MOVIE_BGM",
      "start_time_ms": 2000,
      "duration_ms": 9640,
      "weight_db": -12.0
    }}
  ]
}}
"""
    ),
)


gemini_decide_sliding_window0_prompt = PromptTemplate(
    input_variables=[
        "visual_story",
        "bgm_candidates",
        "video_duration_ms",
        "window_end_ms",
    ],
    template=(
        """
You are a Master Sound Designer scoring ONLY THE FIRST SLICE of a longer video.
The ATTACHED VIDEO is [0, {window_end_ms}] ms of the original. It is ground truth for sync.
The full film is longer: total duration (ms) = {video_duration_ms}.
Score ONLY this slice. Do NOT invent events past {window_end_ms}.

IMPORTANT: Do NOT include any NARRATOR cues. Score with SFX, AMBIENCE, MUSIC, MOVIE_BGM only.
Every cue must satisfy 0 <= start_time_ms and start_time_ms + duration_ms <= {window_end_ms}.
start_time_ms is on the ORIGINAL timeline (this slice starts at 0, so that matches the clip).

═══════════════════════════════════════
STORY CONTEXT (compact; video is primary)
═══════════════════════════════════════
{visual_story}

═══════════════════════════════════════
MOVIE BGM CANDIDATES
═══════════════════════════════════════
Each candidate includes clip_id, duration_ms, similarity, and what the music sounds like.
Include MOVIE_BGM ONLY if it fits this slice. audio_class MUST be the exact clip_id.
Prefer generative MUSIC over a bad MOVIE_BGM. You may omit ALL candidates.
{bgm_candidates}

═══════════════════════════════════════
AUDIO_CLASS / TYPES / MIX
═══════════════════════════════════════
audio_class feeds a generative model. For continuous sounds prefix "CONTINUOUS: " and be specific.
* SFX — hits or continuous beds. AMBIENCE — environment bed. MUSIC — generative score.
* MOVIE_BGM — library only; duration_ms near the candidate's native duration (±10–15%).
DURATION CAP: the generative model cannot render more than 20000 ms in one pass.
Prefer duration_ms <= 20000 for SFX, AMBIENCE, and MOVIE_BGM.
MUSIC may be longer than 20000 ms, and any cue may run past 20000 ms when a cut would break continuity.
weight_db: SFX -2 to -6 (peaks may be hotter); MUSIC/MOVIE_BGM -10 to -15; AMBIENCE -18 to -24.
Lean scoring. Do NOT use word_index.

═══════════════════════════════════════
OUTPUT
═══════════════════════════════════════
JSON ONLY. No markdown. NEVER audio_type "NARRATOR".
{{
  "total_video_duration_ms": {video_duration_ms},
  "audio_cues": [
    {{
      "audio_class": "CONTINUOUS: close rain on metal, mid-field, NOT distant crowd",
      "audio_type": "AMBIENCE",
      "start_time_ms": 0,
      "duration_ms": {window_end_ms},
      "weight_db": -20.0
    }}
  ]
}}
"""
    ),
)


gemini_decide_sliding_next_window_prompt = PromptTemplate(
    input_variables=[
        "visual_story",
        "bgm_candidates",
        "video_duration_ms",
        "window_start_ms",
        "window_end_ms",
        "clip_origin_ms",
        "clip_duration_ms",
        "running_cues",
    ],
    template=(
        """
You are a Master Sound Designer continuing a score into the NEXT slice of a longer video.
The ATTACHED VIDEO is a CONTEXT CLIP. It is NOT a new timeline.

═══════════════════════════════════════
CLOCK (CRITICAL)
═══════════════════════════════════════
* Full video duration (ms): {video_duration_ms}
* Score NEW cues ONLY inside [{window_start_ms}, {window_end_ms}] on the ORIGINAL timeline.
* Attached clip starts at ORIGINAL ms = {clip_origin_ms} (about 5s of lookback before the window).
* Attached clip duration (ms): {clip_duration_ms}
* Do NOT reset the clock to 0. A cue at the window start uses start_time_ms ≈ {window_start_ms}.

The lookback may already have sounds RUNNING. Those cues are listed below.

═══════════════════════════════════════
RUNNING CUES (audible in the lookback)
═══════════════════════════════════════
For EACH running cue, decide whether it should CONTINUE into this window or TERMINATE
at or before {window_start_ms}.
* terminate: the sound should stop by the window boundary (or earlier in the lookback).
  Set end_time_ms <= {window_start_ms}.
* continue: the sound should keep going into this window. Set end_time_ms on the ORIGINAL
  clock, >= {window_start_ms}, and <= {video_duration_ms}. Extend duration if the bed
  still matches the picture.
Do NOT invent new ids. Do NOT omit a running cue — answer continue or terminate for each.
{running_cues}

═══════════════════════════════════════
STORY CONTEXT
═══════════════════════════════════════
{visual_story}

═══════════════════════════════════════
MOVIE BGM CANDIDATES (same library shortlist as the first window)
═══════════════════════════════════════
Include a NEW MOVIE_BGM only if it fits THIS window and is not already covered by a
continued cue. audio_class MUST be the exact clip_id. You may omit all candidates.
{bgm_candidates}

═══════════════════════════════════════
NEW CUES
═══════════════════════════════════════
Add sounds that earn their place in [{window_start_ms}, {window_end_ms}] only.
Do NOT duplicate a running cue you chose to continue.
No NARRATOR. Continuous sounds: prefix audio_class with "CONTINUOUS: ".
DURATION CAP: prefer new cues at or under 20000 ms. MUSIC may be longer, and a cue may
run past 20000 ms when continuity requires one unbroken bed. The model cannot render
more than 20000 ms in one pass.
weight_db: SFX -2 to -6; MUSIC/MOVIE_BGM -10 to -15; AMBIENCE -18 to -24.

═══════════════════════════════════════
OUTPUT (JSON ONLY)
═══════════════════════════════════════
{{
  "cue_continuations": [
    {{ "id": 4, "action": "continue", "end_time_ms": {window_end_ms} }},
    {{ "id": 7, "action": "terminate", "end_time_ms": {window_start_ms} }}
  ],
  "audio_cues": [
    {{
      "audio_class": "close door latch, metal, single hit, NOT a slam",
      "audio_type": "SFX",
      "start_time_ms": {window_start_ms},
      "duration_ms": 800,
      "weight_db": -5.0
    }}
  ]
}}
"""
    ),
)


gemini_decide_more_cues_for_region_prompt = PromptTemplate(
    input_variables=[
        "visual_story",
        "region_prompt",
        "existing_cues",
        "bgm_candidates",
        "selection_start_ms",
        "selection_end_ms",
        "clip_origin_ms",
        "clip_duration_ms",
        "video_duration_ms",
    ],
    template=(
        """
You are a Master Sound Designer ADDING MORE cues to an existing score for a SELECTED REGION.
The ATTACHED VIDEO is a CONTEXT CLIP cut from the original timeline — it is NOT a new timeline.

═══════════════════════════════════════
CLOCK / COORDINATE SYSTEM (CRITICAL)
═══════════════════════════════════════
* Original video duration (ms): {video_duration_ms}
* Selection window (ORIGINAL ms): [{selection_start_ms}, {selection_end_ms}]
  → Generate NEW cues ONLY for this selection window.
* Attached clip starts at ORIGINAL ms = {clip_origin_ms}
* Attached clip duration (ms): {clip_duration_ms}
* Every start_time_ms you output MUST be on the ORIGINAL timeline
  (e.g. if selection starts at 3000, a cue at the start of selection uses start_time_ms ≈ 3000,
  NOT 0). Do NOT reset the clock to the clip start.

═══════════════════════════════════════
REGION INTENT (user request for this fill; may be empty)
═══════════════════════════════════════
{region_prompt}

═══════════════════════════════════════
STORY CONTEXT (compact; video is primary)
═══════════════════════════════════════
{visual_story}

═══════════════════════════════════════
EXISTING CUES IN / OVERLAPPING THE SELECTION
═══════════════════════════════════════
These cues ALREADY exist. Do NOT duplicate them. Fill gaps or add complementary layers
only when useful. Prefer SFX and light beds over restacking MUSIC/AMBIENCE that already
covers the window unless the region intent explicitly asks.
Cues with locked=true are FIXED ANCHORS — never duplicate their beds; accompany them.
Prefer new cues at or under 20000 ms. MUSIC may be longer, and a cue may run past 20000 ms
when continuity requires one unbroken bed. The generative model cannot render more than 20000 ms in one pass.
{existing_cues}

═══════════════════════════════════════
MOVIE BGM CANDIDATES
═══════════════════════════════════════
Include MOVIE_BGM ONLY if musical character fits AND the region intent needs music.
Prefer generative MUSIC over a bad MOVIE_BGM. You may omit ALL candidates.
{bgm_candidates}

═══════════════════════════════════════
AUDIO_CLASS PRECISION (CRITICAL FOR GENERATIVE SFX)
═══════════════════════════════════════
audio_class is fed to a generative audio model. Vague prompts produce wrong sounds.

For ANY continuous / looping / sustained sound:
* Prefix audio_class with "CONTINUOUS: "
* Name the exact source, rate/mechanism, spatial perspective
* Explicitly NEGATE wrong characters
* Set duration_ms to cover the FULL needed span inside the selection

For one-shot SFX: be specific (material, size, force, space).

═══════════════════════════════════════
CUE TYPES (NO NARRATOR)
═══════════════════════════════════════
* SFX — action punctuation OR continuous action beds
* AMBIENCE — environment bed (only if missing / region asks)
* MUSIC — generative score (only if missing / region asks)
* MOVIE_BGM — library track; audio_class MUST be exact candidate clip_id

═══════════════════════════════════════
TIMING RULES
═══════════════════════════════════════
* start_time_ms / duration_ms on the ORIGINAL timeline.
* Every new cue MUST intersect [{selection_start_ms}, {selection_end_ms}].
* Clamp so start_time_ms + duration_ms <= {video_duration_ms}.
* Lean scoring — only cues that earn their place for the region intent.
* Do NOT use word_index.

═══════════════════════════════════════
MIX HIERARCHY
═══════════════════════════════════════
weight_db:
1. SFX: -2.0 to -6.0 dB (explosions may peak +2.0 to +4.0)
2. MUSIC / MOVIE_BGM: -10.0 to -15.0 dB
3. AMBIENCE: -18.0 to -24.0 dB floor

═══════════════════════════════════════
OUTPUT
═══════════════════════════════════════
* JSON ONLY. No markdown fences, no commentary.
* NEVER output audio_type "NARRATOR".
* Return ONLY the NEW cues (not the existing ones).
* Include total_video_duration_ms = {video_duration_ms}.

EXAMPLE:
{{
  "total_video_duration_ms": {video_duration_ms},
  "audio_cues": [
    {{
      "audio_class": "CONTINUOUS: close metallic footsteps on wet asphalt, mid-field, NOT distant crowd",
      "audio_type": "SFX",
      "start_time_ms": {selection_start_ms},
      "duration_ms": 2500,
      "weight_db": -5.0
    }}
  ]
}}
"""
    ),
)


gemini_refine_cues_for_region_prompt = PromptTemplate(
    input_variables=[
        "visual_story",
        "region_prompt",
        "existing_cues",
        "selection_start_ms",
        "selection_end_ms",
        "clip_origin_ms",
        "clip_duration_ms",
        "video_duration_ms",
    ],
    template=(
        """
You are a Master Sound Designer REFINING an existing score for a SELECTED REGION.
The ATTACHED VIDEO is a CONTEXT CLIP cut from the original timeline — it is NOT a new timeline.

═══════════════════════════════════════
CLOCK / COORDINATE SYSTEM (CRITICAL)
═══════════════════════════════════════
* Original video duration (ms): {video_duration_ms}
* Selection window (ORIGINAL ms): [{selection_start_ms}, {selection_end_ms}]
  → Only add/edit/delete cues that belong to this window.
* Attached clip starts at ORIGINAL ms = {clip_origin_ms}
* Attached clip duration (ms): {clip_duration_ms}
* Every start_time_ms MUST be on the ORIGINAL timeline (NOT clip-relative).

═══════════════════════════════════════
REGION INTENT (optional user refine request)
═══════════════════════════════════════
{region_prompt}
If empty: refine for picture sync, balance, and lean cinematic scoring in this window.

═══════════════════════════════════════
STORY CONTEXT
═══════════════════════════════════════
{visual_story}

═══════════════════════════════════════
EXISTING CUES (metadata only; locked=true are FIXED ANCHORS)
═══════════════════════════════════════
{existing_cues}

LOCKED RULES (HARD):
* Never edit, retimed, re-prompt, or delete a cue with locked=true.
* Treat locked cues as fixed — reshape OTHER cues to accompany them.

ALLOWED OPS on unlocked cues:
* Edit timing (start_time_ms, duration_ms), loudness (weight_db), fades, ducking_priority
* Change audio_class / audio_type / narrator text (requires regeneration)
* Delete sparse / conflicting unlocked cues via deleted_cue_ids
* Add NEW complementary cues (added_cues) that intersect the selection

Do NOT invent deletes by omission — list deletes explicitly in deleted_cue_ids.
Do NOT restack MUSIC/AMBIENCE unless the intent asks. Prefer lean scoring.
NEVER output audio_type "NARRATOR" in added_cues for this visual refine path.

═══════════════════════════════════════
OUTPUT (JSON ONLY)
═══════════════════════════════════════
{{
  "added_cues": [
    {{
      "audio_class": "...",
      "audio_type": "SFX",
      "start_time_ms": {selection_start_ms},
      "duration_ms": 2000,
      "weight_db": -5.0
    }}
  ],
  "edited_cues": [
    {{
      "id": 12,
      "needs_regeneration": false,
      "start_time_ms": 3200,
      "duration_ms": 1800,
      "weight_db": -6.0,
      "changed_fields": ["start_time_ms", "duration_ms", "weight_db"],
      "notes": "tighten to hit"
    }}
  ],
  "deleted_cue_ids": [7]
}}

Rules for edited_cues:
* id MUST match an existing unlocked cue.
* Include only fields you change (plus id).
* Set needs_regeneration=true ONLY when audio_class, audio_type, story, or
  narrator_description meaningfully changes. Timing/loudness/fades alone → false.
* changed_fields lists which keys you changed.

Rules for added_cues:
* Full cue dicts on the ORIGINAL clock; must intersect the selection.
* Do not reuse ids of existing cues.
"""
    ),
)


# Shared contract for both sync prompts. `weight_db` is creative intent only:
# loudness calibration against the measured clip level happens in
# superimposition_model/sync_audio_cues.py, so asking the model to also
# compensate for a quiet/loud clip would double-correct every cue.
_SYNC_OUTPUT_CONTRACT = """
Return STRICT JSON only (no prose, no markdown fences), in exactly this shape:

{{
  "audio_cues": [
    {{
      "id": <the SAME integer id from the input cue>,
      "start_time_ms": <integer, >= 0>,
      "duration_ms": <integer, > 0>,
      "weight_db": <float, -45.0 .. 0.0>,
      "ducking_priority": <integer 1..4>,
      "fade_in_ms": <integer, optional>,
      "fade_out_ms": <integer, optional>,
      "notes": "<max 12 words explaining the change, optional>"
    }}
  ]
}}

Hard rules:
- Return EVERY input cue exactly once, keeping its `id`. Never invent or drop cues.
- Never change a cue's audio_type or audio_class; you are only re-timing and re-balancing.
- `weight_db` is the MIX INTENT for this cue relative to other cues. Do NOT try to
  compensate for the clip's own recorded level — an automatic loudness calibration
  stage already normalizes each clip against a per-type loudness target. Use the
  measured values only to judge whether a clip is unusable (silent), abnormally
  peaky, or speech-like, and to reason about relative balance.
- `ducking_priority` is 1..4 where LOWER wins: 1 = speech/narration (must stay
  intelligible, ducks everything else), 2 = key story SFX, 3 = music / BGM,
  4 = background ambience.
- Keep every cue inside the timeline: start_time_ms + duration_ms <= {total_duration_ms}.
- Prefer small, purposeful corrections over wholesale rewrites. Leave a cue's
  numbers unchanged when they are already right.
- Try to keep the duration of the cues as close to the original as possible.
- Speech must never be buried: any music/ambience overlapping narration should sit
  clearly below it.
"""


sync_audio_cues_textual_prompt = PromptTemplate(
    input_variables=[
        "story_text",
        "total_duration_ms",
        "audio_cues",
        "loudness_measurements",
        "whisper_json",
    ],
    template=(
        """
You are a cinematic re-recording mixer doing a final sync pass on an already
generated soundtrack. The clips exist; your job is to place them accurately in
time and balance them against each other.

Story text:
{story_text}

Timeline length: {total_duration_ms} ms

Current audio cues (as placed by the cue decider):
{audio_cues}

Measured loudness of each generated clip (dBFS / LUFS; `lufs` is the gated
integrated loudness of the RAW clip, before any mix gain; `speech_band_ratio` is
the share of energy in the 300-3400 Hz intelligibility band):
{loudness_measurements}

Narrator word timestamps from Whisper (seconds; empty list when there is no
narration). Use these to anchor cues to the exact words they illustrate:
{whisper_json}

Think about:
- Does each cue start when the event it represents actually happens in the narration?
- Does its duration cover the event without smothering the next beat?
- Is the relative balance cinematic (narration intelligible, SFX punchy but not
  overbearing, music/ambience supportive)?
"""
        + _SYNC_OUTPUT_CONTRACT
    ),
)


sync_audio_cues_visual_prompt = PromptTemplate(
    input_variables=[
        "video_content",
        "total_duration_ms",
        "audio_cues",
        "loudness_measurements",
    ],
    template=(
        """
You are a cinematic re-recording mixer doing a final sync pass on an already
generated soundtrack for the attached video. The clips exist; your job is to lock
them to the picture and balance them against each other.

Watch the video and use its actual cut points, motion and on-screen events as the
ground truth for timing.

Timestamped description of the video:
{video_content}

Video / timeline length: {total_duration_ms} ms

Current audio cues (as placed by the cue decider):
{audio_cues}

Measured loudness of each generated clip (dBFS / LUFS; `lufs` is the gated
integrated loudness of the RAW clip, before any mix gain; `speech_band_ratio` is
the share of energy in the 300-3400 Hz intelligibility band):
{loudness_measurements}

Think about:
- Does each cue hit on the frame where the action happens (impact, cut, door, step)?
- Do sustained beds (music, ambience) span whole scenes instead of fragments?
- Is the relative balance cinematic, with dialogue/narration always intelligible?
"""
        + _SYNC_OUTPUT_CONTRACT
    ),
)
