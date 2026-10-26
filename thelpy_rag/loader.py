"""Wczytywanie transkrypcji sesji z plików JSON.

Oczekiwany format (jeden plik = jedna sesja):

{
  "session_id": "S001",
  "client_id": "C017",
  "session_at": "2026-03-14T10:00:00+01:00",
  "duration_seconds": 3120,          # opcjonalne
  "source_version": "v1",            # opcjonalne
  "turns": [
    {"turn_id": "T001", "speaker_id": "SPK0", "speaker_role": "therapist",
     "text": "...", "start": 0.0, "end": 4.2}
  ]
}

Jeśli Twoje pliki wyglądają inaczej, zmień tylko funkcję `session_from_dict`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import Session, Turn

REQUIRED = ("session_id", "client_id", "session_at", "turns")


def session_from_dict(d: dict) -> Session:
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"Brak pól {missing} w sesji {d.get('session_id', '?')}")

    turns = []
    for i, t in enumerate(d["turns"]):
        text = (t.get("text") or "").strip()
        if not text:
            continue
        turns.append(Turn(
            turn_id=t.get("turn_id") or f"T{i + 1:03d}",
            speaker_id=str(t.get("speaker_id", "UNK")),
            speaker_role=t.get("speaker_role"),
            text=text,
            start=t.get("start"),
            end=t.get("end"),
        ))
    if not turns:
        raise ValueError(f"Sesja {d['session_id']} nie ma żadnych niepustych tur")

    return Session(
        session_id=str(d["session_id"]),
        client_id=str(d["client_id"]),
        session_at=d["session_at"],
        added_at=d.get("added_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=d.get("duration_seconds"),
        source_version=str(d.get("source_version", "v1")),
        turns=turns,
    )


def load_session(path: str | Path) -> Session:
    with open(path, encoding="utf-8") as f:
        return session_from_dict(json.load(f))


def iter_session_files(paths: list[str]) -> list[Path]:
    """Przyjmuje pliki i/lub katalogi; z katalogów bierze *.json."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.suffix == ".json":
            out.append(p)
    return out
