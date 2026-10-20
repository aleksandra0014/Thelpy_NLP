#!/usr/bin/env python3
"""
Simple script for testing prompts against a local Ollama server.

Sends a therapy session transcript + a prompt to a model (e.g. qwen3:8b)
via the local Ollama API and saves the result (prompt, params, response)
to a JSON file under logs/ -- the start of an experiment log for the thesis.

Supports two transcript formats:
  - .txt  : plain text, used as-is
  - .json : structured utterances, converted to a readable transcript.
            Accepts either a bare list of utterances, or an object with
            an "utterances" / "turns" / "transcript" key holding that list.
            Each utterance is matched flexibly for common field names:
              speaker: "speaker", "role", "author", "who"
              text:    "text", "utterance", "content", "value"
              time:    "timestamp", "start", "time" (optional)

Usage examples:
    python run_prompt.py --transcript sample_transcript.txt --task summary

    python run_prompt.py --transcript sample_transcript.json --task summary

    python run_prompt.py \
        --transcript sample_transcript.txt \
        --prompt "List all cognitive distortions visible in this conversation." \
        --model qwen3:8b \
        --num-ctx 8192
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Defaults ---------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_NUM_CTX = 8192  # set explicitly so Ollama doesn't silently truncate context
DEFAULT_TEMPERATURE = 0.3  # lower temperature = more deterministic, repeatable tests

LOGS_DIR = Path(__file__).parent / "logs"

# Possible key names for each field, tried in order, to stay tolerant of
# different annotation schemas (e.g. AnnoMI-style exports vs your own JSON).
SPEAKER_KEYS = ["speaker", "role", "author", "who", "speaker_name"]
TEXT_KEYS = ["text", "utterance", "content", "value", "transcript", "utterance_text"]
TIME_KEYS = ["timestamp", "start", "time", "start_time"]

# Ready-made prompt templates for the tasks from the thesis.
# Extend these as you keep testing.
TASK_PROMPTS = {
    "summary": (
        "You are an assistant supporting a therapist. Below is a transcript of "
        "part of a therapy session. Summarize this conversation in 5-7 sentences, "
        "focusing on: the main topic raised by the client, their current emotional "
        "state, and the therapist's interventions. Write factually, keep it "
        "confidential and professional in tone.\n\n"
        "Transcript:\n{transcript}\n\nSummary:"
    ),
    "patient_description": (
        "Based on the transcript below, prepare a short client description "
        "(3-5 sentences) useful as a note before the next session. Include: the "
        "presenting problem, visible thought or behavior patterns, and the "
        "emotional tone of the session. Do not infer diagnoses; stick to what "
        "follows directly from the conversation.\n\n"
        "Transcript:\n{transcript}\n\nClient description:"
    ),
    "case_conceptualization": (
        "You are an assistant to a therapist working within a cognitive-"
        "behavioral framework. Based on the transcript below, propose a "
        "preliminary case conceptualization in this format:\n"
        "- Situation/trigger:\n- Automatic thoughts:\n- Emotions:\n- Behaviors:\n"
        "- Possible intermediate beliefs:\n\n"
        "Base this only on what appears in the transcript, and note that it is a "
        "preliminary hypothesis for the therapist to verify.\n\n"
        "Transcript:\n{transcript}\n\nConceptualization:"
    ),
    "supervision_report": (
        "Prepare a short report on this session excerpt for supervision purposes. "
        "Include: the session topic, therapeutic techniques used, the client's "
        "response, and one question or concern the therapist might raise during "
        "supervision.\n\n"
        "Transcript:\n{transcript}\n\nReport:"
    ),
    "cognitive_distortions": (
        "Analyze the transcript below and list, as bullet points, the cognitive "
        "distortions visible in the client's statements (e.g. catastrophizing, "
        "black-and-white thinking, mind reading). For each one, give a short "
        "quote from the transcript as evidence. If you find none, say so "
        "explicitly instead of guessing.\n\n"
        "Transcript:\n{transcript}\n\nCognitive distortions:"
    ),
}


def _first_present(d: dict, keys: list[str]):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _utterances_from_json(data) -> list:
    """Extract a list of utterance dicts from various JSON shapes."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("utterances", "turns", "transcript", "messages", "dialogue", "dialog"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(
        "Could not find a list of utterances in the JSON file. Expected either "
        "a bare list, or an object with an 'utterances'/'turns'/'transcript'/'dialog' key."
    )


# Top-level metadata fields worth surfacing as context before the dialogue,
# e.g. AnnoMI-style exports carry "topic", "mi_quality", "video_title".
METADATA_LABELS = {
    "topic": "Topic",
    "video_title": "Title",
    "mi_quality": "MI quality",
    "transcript_id": "Transcript ID",
    "session_id": "Session ID",
}


def _extract_metadata_header(data: dict) -> str:
    """Build a short metadata header from recognized top-level fields, if present."""
    lines = []
    for key, label in METADATA_LABELS.items():
        if key in data and data[key] not in (None, ""):
            lines.append(f"{label}: {data[key]}")
    if not lines:
        return ""
    return "Session metadata:\n" + "\n".join(lines) + "\n\n"


def load_transcript(path: Path) -> str:
    """Load a transcript from .txt (as-is) or .json (flattened to readable text)."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        utterances = _utterances_from_json(raw)

        metadata_header = _extract_metadata_header(raw) if isinstance(raw, dict) else ""

        lines = []
        for i, u in enumerate(utterances):
            if not isinstance(u, dict):
                # tolerate a plain list of strings too
                lines.append(str(u))
                continue

            speaker = _first_present(u, SPEAKER_KEYS) or "Unknown"
            text = _first_present(u, TEXT_KEYS)
            timestamp = _first_present(u, TIME_KEYS)

            if text is None:
                raise ValueError(
                    f"Utterance #{i} has no recognizable text field. "
                    f"Tried keys: {TEXT_KEYS}. Got keys: {list(u.keys())}"
                )

            prefix = f"[{timestamp}] " if timestamp is not None else ""
            lines.append(f"{prefix}{speaker}: {text}")

        return metadata_header + "\n".join(lines)

    raise ValueError(f"Unsupported transcript format: '{suffix}'. Use .txt or .json.")


def build_prompt(task: str | None, custom_prompt: str | None, transcript: str) -> str:
    if custom_prompt:
        if "{transcript}" in custom_prompt:
            return custom_prompt.format(transcript=transcript)
        return f"{custom_prompt}\n\nTranscript:\n{transcript}"

    if task not in TASK_PROMPTS:
        available = ", ".join(TASK_PROMPTS.keys())
        raise ValueError(f"Unknown task '{task}'. Available: {available}")

    return TASK_PROMPTS[task].format(transcript=transcript)


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    num_ctx: int,
    temperature: float,
    timeout: int = 300,
) -> dict:
    """Call the local Ollama API (/api/generate) and return the full response."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }

    start = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - start

    response.raise_for_status()
    data = response.json()
    data["_elapsed_seconds"] = round(elapsed, 2)
    return data


def save_log(
    task: str | None,
    prompt: str,
    transcript_path: str,
    model: str,
    num_ctx: int,
    temperature: float,
    result: dict,
) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = LOGS_DIR / f"{timestamp}_{task or 'custom'}.json"

    log_entry = {
        "timestamp_utc": timestamp,
        "model": model,
        "task": task,
        "transcript_file": transcript_path,
        "params": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
        "prompt": prompt,
        "response": result.get("response", ""),
        "elapsed_seconds": result.get("_elapsed_seconds"),
        "eval_count": result.get("eval_count"),  # number of generated tokens
        "eval_duration_ns": result.get("eval_duration"),  # for tokens/sec
    }

    out_path.write_text(json.dumps(log_entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Test prompts against a local model via Ollama.")
    parser.add_argument("--transcript", required=True, help="Path to the transcript file (.txt or .json)")
    parser.add_argument(
        "--task",
        choices=list(TASK_PROMPTS.keys()),
        help=f"Ready-made task template: {', '.join(TASK_PROMPTS.keys())}",
    )
    parser.add_argument("--prompt", help="Custom prompt (overrides --task). Use {transcript} as a placeholder.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name in Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"Context window size (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--show-transcript", action="store_true", help="Print the parsed transcript before sending (useful for debugging JSON parsing)")

    args = parser.parse_args()

    if not args.task and not args.prompt:
        parser.error("Provide --task (ready-made template) or --prompt (custom one).")

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    try:
        transcript = load_transcript(transcript_path)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error while parsing transcript: {e}", file=sys.stderr)
        sys.exit(1)

    if args.show_transcript:
        print("--- PARSED TRANSCRIPT ---")
        print(transcript)
        print("--- END TRANSCRIPT ---\n")

    prompt = build_prompt(args.task, args.prompt, transcript)

    print(f"-> Model: {args.model} | num_ctx={args.num_ctx} | temperature={args.temperature}")
    print(f"-> Sending request to {args.url} ...")

    try:
        result = call_ollama(args.url, args.model, prompt, args.num_ctx, args.temperature)
    except requests.exceptions.ConnectionError:
        print(
            "Error: could not connect to Ollama. Check that the service is running "
            "('ollama serve' or 'systemctl status ollama') and that --url is correct.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}\nServer response: {e.response.text}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- MODEL RESPONSE ({result.get('_elapsed_seconds')}s) ---\n")
    print(result.get("response", "(no response)"))
    print("\n--- END ---\n")

    log_path = save_log(
        task=args.task,
        prompt=prompt,
        transcript_path=str(transcript_path),
        model=args.model,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        result=result,
    )
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()