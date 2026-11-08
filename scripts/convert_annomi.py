"""Konwersja transkrypcji AnnoMI (format surowy) do formatu sesji thelpy_rag (jak data/annomi_64.json).

Wejście (AnnoMI):
    {"transcript_id": 5, "video_title": ..., "topic": ..., "mi_quality": ...,
     "dialog": [{"timestamp": "00:25:24", "speaker": "therapist", "text": "..."}, ...]}

Wyjście (thelpy_rag/loader.py):
    {"session_id": 5, "session_at": ..., "patient_id": 2, "added_at": ..., "duration_seconds": 747.0,
     "turns": [{"turn_id": "T001", "speaker_id": "SPK0", "speaker_role": "therapist", "text": "...",
                "start": "00:25:24", "end": "00:25:24", "audio_features": {... null ...}}, ...]}

Reguły (odtworzone z data/annomi_64.json):
- turn_id: T001, T002, ... w kolejności dialogu,
- therapist -> SPK0, patient/client -> SPK1,
- start = timestamp tury; end = timestamp następnej tury (jeśli następna ma wcześniejszy znacznik
  czasu — zdarza się w AnnoMI — end = start); ostatnia tura: end = start + 1 s,
- duration_seconds = end ostatniej tury - start pierwszej,
- audio_features: wszystkie cechy null (transkrypcja tekstowa, bez audio).

    python -m scripts.convert_annomi data/raw/annomi_5.json -o data/annomi_5.json --patient-id 2
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from thelpy_rag.loader import session_from_dict, to_seconds

SPEAKERS = {"therapist": "SPK0", "patient": "SPK1", "client": "SPK1"}
AUDIO_FEATURES = ["f0_mean", "f0_var", "loudness_mean", "loudness_range", "alpha_ratio", "hammarberg",
                  "jitter", "shimmer", "hnr", "words_per_min", "silence_ratio", "arousal"]


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def convert(raw: dict, patient_id, session_at: str) -> dict:
    dialog = [d for d in raw["dialog"] if (d.get("text") or "").strip()]
    if not dialog:
        raise ValueError(f"Transkrypcja {raw.get('transcript_id')} nie ma niepustych wypowiedzi")
    starts = [to_seconds(d["timestamp"]) for d in dialog]

    turns = []
    for i, d in enumerate(dialog):
        role = d["speaker"].lower()
        if role not in SPEAKERS:
            raise ValueError(f"Nieznany mówiący '{d['speaker']}' w wypowiedzi {i + 1}")
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(dialog) else start + 1
        end = max(end, start)
        turns.append({
            "turn_id": f"T{i + 1:03d}",
            "speaker_id": SPEAKERS[role],
            "speaker_role": "patient" if role == "client" else role,
            "text": d["text"],
            "start": hms(start),
            "end": hms(end),
            "audio_features": dict.fromkeys(AUDIO_FEATURES),
        })

    duration = to_seconds(turns[-1]["end"]) - to_seconds(turns[0]["start"])
    return {
        "session_id": raw["transcript_id"],
        "session_at": session_at,
        "patient_id": patient_id,
        "added_at": session_at,
        "duration_seconds": float(duration),
        "turns": turns,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="plik AnnoMI (.json z kluczem 'dialog')")
    ap.add_argument("-o", "--output", required=True, help="plik wyjściowy")
    ap.add_argument("--patient-id", type=int, required=True, help="ID pacjenta (różni pacjenci = różne ID)")
    ap.add_argument("--session-at", default=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    help="data sesji ISO 8601 (domyślnie teraz, UTC)")
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = convert(raw, args.patient_id, args.session_at)
    session_from_dict(out)   # walidacja formatem, który czyta thelpy_rag
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.input} -> {args.output}: sesja {out['session_id']}, {len(out['turns'])} tur, "
          f"{out['duration_seconds']:.0f} s, patient_id={out['patient_id']}")


if __name__ == "__main__":
    main()
