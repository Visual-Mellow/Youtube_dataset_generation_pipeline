"""Structured pipeline UI logs: JSONL in a ``.log`` file + HTML viewer.

Each ``pipeline.ui.log`` begins with a magic header line, then one JSON object
per line. Render with :func:`render_html` or ``python tools/view_ui_log.py``.
"""
from __future__ import annotations

import html
import json
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

UI_LOG_MAGIC = "# VisualMellow-UI-LOG-v1"
UI_LOG_VERSION = 1
_MAX_STR = 120_000
_MAX_LIST = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"_type": "bytes", "size": len(value), "note": "binary omitted from log"}
    if isinstance(value, str):
        if len(value) <= _MAX_STR:
            return value
        return value[:_MAX_STR] + f"\n… ({len(value) - _MAX_STR} more chars)"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:_MAX_LIST]:
            out[str(k)] = _sanitize(v, depth=depth + 1)
        if len(value) > _MAX_LIST:
            out["…"] = f"({len(value) - _MAX_LIST} more keys)"
        return out
    if isinstance(value, (list, tuple)):
        items = [_sanitize(v, depth=depth + 1) for v in value[:_MAX_LIST]]
        if len(value) > _MAX_LIST:
            items.append(f"… ({len(value) - _MAX_LIST} more items)")
        return items
    return str(value)


class UILogWriter:
    """Append structured events to a ``*.ui.log`` file (JSONL + magic header)."""

    def __init__(self, path: Path, *, fresh: bool = True, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open_steps: dict[str, float] = {}
        if fresh or not self.path.is_file():
            self.path.write_text(UI_LOG_MAGIC + "\n", encoding="utf-8")
            self.emit("session_start", title="Pipeline session", data=meta or {})
        elif meta:
            self.emit("session_resume", title="Pipeline session (append)", data=meta)

    def emit(self, kind: str, *, title: str = "", step_id: str | None = None, data: Any = None) -> None:
        row = {
            "v": UI_LOG_VERSION,
            "ts": _utc_now(),
            "kind": kind,
            "title": title,
            "step_id": step_id,
            "data": _sanitize(data) if data is not None else None,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def message(self, text: str, *, step_id: str | None = None) -> None:
        self.emit("message", title=text, step_id=step_id, data={"text": text})

    def step_begin(self, step_id: str, title: str, *, inputs: Any = None) -> None:
        import time

        self._open_steps[step_id] = time.perf_counter()
        self.emit("step_begin", title=title, step_id=step_id, data={"inputs": inputs})

    def step_end(self, step_id: str, summary: str = "", *, outputs: Any = None) -> None:
        import time

        t0 = self._open_steps.pop(step_id, None)
        elapsed = round(time.perf_counter() - t0, 3) if t0 is not None else None
        self.emit(
            "step_end",
            title=summary or step_id,
            step_id=step_id,
            data={"outputs": outputs, "elapsed_s": elapsed},
        )

    def data_block(self, step_id: str, role: str, label: str, payload: Any) -> None:
        self.emit(
            "data",
            title=label,
            step_id=step_id,
            data={"role": role, "label": label, "payload": payload},
        )

    def llm_call(
        self,
        step_id: str,
        *,
        stage: str,
        model: str,
        prompt: str,
        result: Any,
        raw_text: str | None = None,
        latency_s: float | None = None,
        video_bytes: int = 0,
        audio_bytes: int = 0,
        mime_type: str = "",
        raw_file: str | None = None,
        error: str | None = None,
    ) -> None:
        self.emit(
            "llm_call",
            title=f"LLM {stage} ({model})",
            step_id=step_id,
            data={
                "stage": stage,
                "model": model,
                "prompt": prompt,
                "response_parsed": result,
                "response_raw_text": raw_text,
                "latency_s": latency_s,
                "attachments": {
                    "video_bytes": video_bytes,
                    "audio_bytes": audio_bytes,
                    "mime_type": mime_type,
                },
                "raw_file": raw_file,
                "error": error,
            },
        )

    def tool_call(self, step_id: str, name: str, *, args: Any = None, result: Any = None) -> None:
        self.emit(
            "tool_call",
            title=f"Tool: {name}",
            step_id=step_id,
            data={"name": name, "args": args, "result": result},
        )

    def file_written(self, step_id: str, path: Path | str, *, detail: Any = None) -> None:
        p = Path(path)
        size = p.stat().st_size if p.is_file() else None
        self.emit(
            "file_write",
            title=f"Wrote {p.name}",
            step_id=step_id,
            data={"path": str(p), "bytes": size, "detail": detail},
        )


def iter_events(log_path: Path) -> Iterator[dict[str, Any]]:
    """Parse a UI log file into event dicts."""
    text = Path(log_path).read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            if i == 0 and line != UI_LOG_MAGIC:
                raise ValueError(f"Not a VisualMellow UI log (expected {UI_LOG_MAGIC!r})")
            continue
        yield json.loads(line)


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def render_html(log_path: Path, *, out_path: Path | None = None, log_dir: Path | None = None) -> Path:
    """Build a standalone collapsible HTML report from a ``pipeline.ui.log``."""
    log_path = Path(log_path).resolve()
    log_dir = log_dir or log_path.parent
    events = list(iter_events(log_path))

    steps: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    session_meta: dict[str, Any] = {}
    loose: list[dict[str, Any]] = []

    for ev in events:
        kind = ev.get("kind") or ""
        sid = ev.get("step_id")
        if kind == "session_start":
            session_meta = ev.get("data") or {}
            continue
        if kind == "step_begin" and sid:
            if sid not in steps:
                steps[sid] = {"title": ev.get("title") or sid, "events": [], "ended": False}
                order.append(sid)
            steps[sid]["events"].append(ev)
            continue
        if sid and sid in steps:
            steps[sid]["events"].append(ev)
            if kind == "step_end":
                steps[sid]["ended"] = True
                if ev.get("data", {}).get("elapsed_s") is not None:
                    steps[sid]["elapsed_s"] = ev["data"]["elapsed_s"]
            continue
        loose.append(ev)

    def block_json(label: str, obj: Any, open_default: bool = False) -> str:
        open_attr = " open" if open_default else ""
        body = html.escape(_json_pretty(obj))
        return (
            f'<details class="block"{open_attr}><summary>{html.escape(label)}</summary>'
            f'<pre class="json">{body}</pre></details>'
        )

    def block_text(label: str, text: str, open_default: bool = False) -> str:
        open_attr = " open" if open_default else ""
        body = html.escape(text)
        return (
            f'<details class="block"{open_attr}><summary>{html.escape(label)}</summary>'
            f'<pre class="text">{body}</pre></details>'
        )

    def render_event(ev: dict[str, Any]) -> str:
        kind = ev.get("kind") or ""
        data = ev.get("data") or {}
        ts = ev.get("ts", "")
        if kind == "message":
            return f'<div class="msg"><span class="ts">{html.escape(ts)}</span> {html.escape(data.get("text", ev.get("title", "")))}</div>'
        if kind == "data":
            role = data.get("role", "data")
            label = data.get("label") or ev.get("title") or role
            payload = data.get("payload")
            css = "in" if role == "input" else "out" if role == "output" else "data"
            return f'<div class="evt {css}">{block_json(f"{label} ({role})", payload)}</div>'
        if kind == "llm_call":
            parts = [
                f'<div class="evt llm"><strong>{html.escape(ev.get("title", "LLM"))}</strong> '
                f'<span class="muted">{html.escape(ts)}</span></div>',
                block_text("Prompt sent to model", data.get("prompt") or "", open_default=False),
            ]
            if data.get("response_raw_text"):
                parts.append(block_text("Raw model text", data["response_raw_text"]))
            if data.get("response_parsed") is not None:
                parts.append(block_json("Parsed response", data["response_parsed"]))
            att = data.get("attachments") or {}
            parts.append(block_json("Attachments / meta", {
                **att,
                "latency_s": data.get("latency_s"),
                "error": data.get("error"),
                "raw_file": data.get("raw_file"),
            }))
            raw_file = data.get("raw_file")
            if raw_file:
                raw_path = log_dir / "llm" / raw_file if not Path(str(raw_file)).is_absolute() else Path(raw_file)
                if not raw_path.is_file():
                    raw_path = log_dir / raw_file
                if raw_path.is_file():
                    try:
                        raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
                        parts.append(block_json(f"Full raw file ({raw_path.name})", raw_doc))
                    except Exception:
                        parts.append(block_text(f"Raw file ({raw_path.name})", raw_path.read_text(encoding="utf-8")[:50000]))
            return "\n".join(parts)
        if kind == "tool_call":
            return (
                f'<div class="evt tool">{block_json(ev.get("title", "Tool"), data)}</div>'
            )
        if kind == "file_write":
            return (
                f'<div class="evt file"><strong>Output file</strong> '
                f'<code>{html.escape(str(data.get("path", "")))}</code> '
                f'({data.get("bytes", "?")} bytes)'
                f'{block_json("Detail", data.get("detail")) if data.get("detail") is not None else ""}'
                f'</div>'
            )
        if kind == "step_end":
            summary = ev.get("title") or "done"
            elapsed = (data or {}).get("elapsed_s")
            extra = f" — {elapsed}s" if elapsed is not None else ""
            outs = (data or {}).get("outputs")
            tail = block_json("Step output", outs) if outs is not None else ""
            return f'<div class="step-end">✓ {html.escape(summary)}{html.escape(extra)}{tail}</div>'
        return block_json(kind, ev)

    step_html_parts: list[str] = []
    for sid in order:
        st = steps[sid]
        elapsed = st.get("elapsed_s")
        badge = f' <span class="badge">{elapsed}s</span>' if elapsed is not None else ""
        inner = "\n".join(render_event(e) for e in st["events"] if e.get("kind") != "step_begin")
        begin = next((e for e in st["events"] if e.get("kind") == "step_begin"), None)
        inputs = (begin or {}).get("data", {}).get("inputs")
        head = block_json("Step inputs", inputs) if inputs is not None else ""
        step_html_parts.append(
            f'<details class="step" open><summary><span class="step-id">{html.escape(sid)}</span> '
            f'{html.escape(st["title"])}{badge}</summary><div class="step-body">{head}{inner}</div></details>'
        )

    loose_html = "\n".join(render_event(e) for e in loose) if loose else ""

    title = session_meta.get("title") or log_path.name
    generated = _utc_now()
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)} — VisualMellow pipeline log</title>
<style>
:root {{ color-scheme: dark light; --bg:#0f1218; --card:#1a1f2b; --border:#2d3548; --text:#e8ecf4; --muted:#9aa3b5; --accent:#6ea8fe; --in:#1e3a2f; --out:#2a2a4a; --llm:#3d2a1e; }}
body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:0; padding:1.5rem; background:var(--bg); color:var(--text); line-height:1.45; }}
h1 {{ font-size:1.25rem; margin:0 0 .25rem; }}
.meta {{ color:var(--muted); font-size:.85rem; margin-bottom:1.25rem; }}
.step {{ background:var(--card); border:1px solid var(--border); border-radius:8px; margin-bottom:.75rem; padding:.5rem .75rem; }}
.step summary {{ cursor:pointer; font-weight:600; list-style-position:outside; }}
.step-id {{ color:var(--accent); font-family: ui-monospace, monospace; font-size:.85em; margin-right:.5rem; }}
.badge {{ font-weight:normal; font-size:.8em; color:var(--muted); }}
.step-body {{ margin-top:.75rem; padding-top:.5rem; border-top:1px solid var(--border); }}
.block {{ margin:.5rem 0; background:rgba(0,0,0,.2); border-radius:6px; padding:.35rem .5rem; }}
.block summary {{ cursor:pointer; font-size:.9rem; color:var(--muted); }}
pre.json, pre.text {{ margin:.5rem 0 0; padding:.75rem; overflow:auto; max-height:28rem; font-size:.78rem; background:#0a0c10; border-radius:4px; white-space:pre-wrap; word-break:break-word; }}
.evt.in .block {{ border-left:3px solid #3dd68c; }}
.evt.out .block {{ border-left:3px solid #a78bfa; }}
.evt.llm {{ border-left:3px solid #f0a060; padding-left:.5rem; margin:.75rem 0; }}
.msg {{ font-size:.9rem; margin:.25rem 0; }}
.ts {{ color:var(--muted); font-size:.75rem; margin-right:.5rem; }}
.muted {{ color:var(--muted); font-size:.8rem; }}
code {{ font-size:.85em; }}
.toolbar {{ margin-bottom:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }}
.toolbar button {{ padding:.35rem .75rem; border-radius:6px; border:1px solid var(--border); background:var(--card); color:var(--text); cursor:pointer; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">Source: <code>{html.escape(str(log_path))}</code><br/>Rendered: {html.escape(generated)}</p>
<div class="toolbar">
<button type="button" onclick="document.querySelectorAll('details.step').forEach(d=>d.open=true)">Expand all steps</button>
<button type="button" onclick="document.querySelectorAll('details.step').forEach(d=>d.open=false)">Collapse all steps</button>
</div>
{block_json("Session metadata", session_meta) if session_meta else ""}
{"".join(step_html_parts)}
{loose_html}
</body>
</html>"""

    out = out_path or log_path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(doc, encoding="utf-8")
    os.replace(tmp, out)
    return out


def open_log_in_browser(log_path: Path, *, html_path: Path | None = None) -> Path:
    html_file = render_html(log_path, out_path=html_path)
    webbrowser.open(html_file.resolve().as_uri())
    return html_file
