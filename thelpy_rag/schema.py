"""Struktury danych: sesja, tura, chunk."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    turn_id: str
    speaker_id: str
    text: str
    speaker_role: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Session:
    session_id: str
    client_id: str
    session_at: str
    turns: list[Turn]
    added_at: Optional[str] = None     
    duration_seconds: Optional[float] = None
    source_version: str = "v1"

    def payload(self) -> dict:
        """Metadane poziomu sesji kopiowane do każdego chunka."""
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "session_at": self.session_at,
            "added_at": self.added_at,
            "duration_seconds": self.duration_seconds,
            "source_version": self.source_version,
        }


@dataclass
class Chunk:
    chunk_id: str
    session_id: str
    chunk_index: int
    strategy: str
    text: str
    turn_ids: list[str]
    speaker_ids: list[str]
    speaker_roles: list[str]
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    n_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "text": self.text,
            "turn_ids": self.turn_ids,
            "turn_range": f"{self.turn_ids[0]}–{self.turn_ids[-1]}" if self.turn_ids else None,
            "speaker_ids": self.speaker_ids,
            "speaker_roles": self.speaker_roles,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "n_tokens": self.n_tokens,
            **self.extra,
        }
