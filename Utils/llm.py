import os
import json
import re
import logging
import time
import tempfile
import dotenv
from typing import Optional

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Inline video payloads above this size use the Files API (more reliable than huge inline parts).
GEMINI_INLINE_VIDEO_MAX_BYTES = 15 * 1024 * 1024


def _repair_truncated_json(json_str: str) -> str:
    """Best-effort close of truncated JSON objects/arrays from small local LLMs."""
    s = json_str.strip()
    if not s:
        return s

    # If we died mid-string, close the quote.
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        s += '"'

    # Drop a trailing dangling comma / colon before closing.
    s = re.sub(r",\s*$", "", s)
    s = re.sub(r":\s*$", ": null", s)

    # Balance braces / brackets.
    opens = s.count("{") - s.count("}")
    opens_arr = s.count("[") - s.count("]")
    if opens > 0:
        s += "}" * opens
    if opens_arr > 0:
        s += "]" * opens_arr
    return s


def _extract_json_fields_loose(text: str) -> dict:
    """Pull string fields even from broken JSON via key:"value" regex."""
    out = {}
    for key, val in re.findall(
        r'"([^"]+)"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    ):
        # Unescape common sequences
        try:
            out[key] = json.loads(f'"{val}"')
        except Exception:
            out[key] = val.replace('\\"', '"').replace("\\n", "\n")
    # Also capture truncated last string: "key": "unterminated...
    for key, val in re.findall(
        r'"([^"]+)"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)',
        text,
        flags=re.DOTALL,
    ):
        if key not in out and val is not None:
            out[key] = val.replace('\\"', '"').replace("\\n", "\n")
    return out


def _parse_json_from_llm_text(response_text: str):
    """Strip markdown fences and extract JSON; repair truncated local-LLM output."""
    if not response_text:
        return None
    json_str = response_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{[\s\S]*", json_str)  # allow unclosed object
    if match:
        # Prefer full object/array if present, else take from first brace.
        full = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", json_str)
        json_str = full.group() if full else match.group()

    for candidate in (json_str, _repair_truncated_json(json_str)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    loose = _extract_json_fields_loose(response_text)
    if loose:
        logger.warning(
            "LLM JSON malformed; recovered %d fields via loose parse: %s",
            len(loose),
            list(loose.keys()),
        )
        return loose

    logger.error("Failed to parse LLM JSON | text=%s", response_text[:500])
    return None


def query_llm(
    llm_name: str,
    model_name: str,
    prompt: str,
    api_key: str = "",
    system_prompt: str = "",
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
):
    """
    Query a supported LLM (gemini, gpt, grok, etc.) and return parsed JSON response.
    llm_name: one of "gemini", "gpt", "grok", "ollama_gemini"
    model_name: model version/name for the target llm
    prompt: user input prompt
    api_key: (optional) API key for selected LLM, overrides env
    system_prompt: (optional) system prompt if supported (e.g. for chatgpt)
    temperature / num_predict: optional overrides for the ollama_gemini branch
      (defaults: 0.1 / 1024).
    Returns:
        dict (parsed JSON object) or None on error.
    """
    llm_name = llm_name.lower()
    result = None
    ollama_temperature = 0.1 if temperature is None else float(temperature)
    ollama_num_predict = 1024 if num_predict is None else int(num_predict)

    if llm_name == "gemini":
        try:
            import google.genai as genai
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                logger.error("Gemini API key not found in environment.")
                return None
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model=model_name, contents=prompt)
            response_text = getattr(response, "text", None)
            if not response_text:
                logger.error("No response text from Gemini.")
                return None
            result = _parse_json_from_llm_text(response_text)
        except Exception as e:
            logger.error(f"Gemini query failed: {e}", exc_info=True)
            return None

    elif llm_name == "gpt":
        try:
            import openai
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                logger.error("OpenAI API key not found in environment.")
                return None
            openai.api_key = key
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
            )
            response_text = completion.choices[0].message.content
            if response_text:
                result = _parse_json_from_llm_text(response_text)
            else:
                logger.error("No response text from ChatGPT.")
                return None
        except Exception as e:
            logger.error(f"ChatGPT query failed: {e}", exc_info=True)
            return None

    elif llm_name == "ollama_gemini":
        try:
            import ollama

            logger.info("Querying local Ollama model=%s", model_name)
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": ollama_temperature,
                    "num_predict": ollama_num_predict,
                },
            )
            response_text = (response or {}).get("message", {}).get("content")
            if not response_text and hasattr(response, "message"):
                response_text = getattr(response.message, "content", None)
            if response_text:
                result = _parse_json_from_llm_text(response_text)
            else:
                logger.error("No response text from local Gemma (Ollama).")
                return None
        except Exception as e:
            logger.error(f"Local Gemma via Ollama query failed: {e}", exc_info=True)
            return None
   
    else:
        logger.error(f"Unsupported LLM name: {llm_name}")
        return None

    return result


def query_gemini_multimodal(
    model_name: str,
    prompt: str,
    *,
    video_bytes: bytes,
    mime_type: str,
    api_key: str = "",
):
    """
    Send text + video to Gemini and return parsed JSON (dict or list).

    Uses inline bytes for smaller payloads; above GEMINI_INLINE_VIDEO_MAX_BYTES
    uploads via the Files API, waits until ACTIVE, then generates.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.error("Gemini API key not found in environment.")
        return None
    try:
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=key)

        def _generate(contents_payload):
            response = client.models.generate_content(
                model=model_name,
                contents=contents_payload,
            )
            response_text = getattr(response, "text", None)
            if not response_text:
                logger.error("No response text from Gemini (multimodal).")
                return None
            return _parse_json_from_llm_text(response_text)

        video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)

        if len(video_bytes) <= GEMINI_INLINE_VIDEO_MAX_BYTES:
            # SDK accepts mixed str + Part in contents list
            return _generate([prompt, video_part])

        # Files API path for large videos
        suffix = ".mp4" if "mp4" in mime_type else ".webm" if "webm" in mime_type else ".bin"
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.write(fd, video_bytes)
            os.close(fd)
            uploaded = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            file_name = getattr(uploaded, "name", None)
            if not file_name:
                logger.error("Gemini file upload did not return a file name.")
                return None
            active = None
            for _ in range(60):
                active = client.files.get(name=file_name)
                state = getattr(active, "state", None)
                state_name = getattr(state, "name", None) if state is not None else str(state)
                if state_name == "ACTIVE":
                    break
                if state_name == "FAILED":
                    logger.error("Gemini file processing FAILED.")
                    return None
                time.sleep(2)
            else:
                logger.error("Timeout waiting for Gemini uploaded file to become ACTIVE.")
                return None
            return _generate([prompt, active])
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"Gemini multimodal query failed: {e}", exc_info=True)
        return None




