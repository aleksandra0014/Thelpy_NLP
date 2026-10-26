"""Strategie chunkingu.

1. fixed_tokens  — baseline: cały transkrypt sklejony w jeden tekst i cięty co `size`
   tokenów (z opcjonalnym overlapem). Granice tur i mówiących są ignorowane, ale
   metadane (turn_ids, mówiący, czasy) są odtwarzane z pozycji znaków.
2. turn_window   — okno N kolejnych tur; żadna wypowiedź nie jest przecinana.
"""
from __future__ import annotations

import re
import warnings
from functools import lru_cache

from . import config
from .schema import Chunk, Session, Turn

ROLE_LABELS = {"therapist": "Terapeuta", "client": "Klient", "patient": "Klient"}


def render_turn(turn: Turn) -> str:
    label = ROLE_LABELS.get((turn.speaker_role or "").lower(), turn.speaker_role or turn.speaker_id)
    return f"{label}: {turn.text}"
# Każda wypowiedź dostaje przed sobą etykietę mówiącego: Terapeuta: Jak Pani reaguje na te telefony od mamy?



# --- tokenizer ---------------------------------------------------------------
class Tokenizer:
    """Tokenizer modelu (HF `tokenizers`) z fallbackiem na podział regexem."""

    def __init__(self, name: str):
        self.name = name
        self._tok = None
        try:
            from tokenizers import Tokenizer as HFTokenizer
            self._tok = HFTokenizer.from_pretrained(name)
        except Exception as e:  # brak sieci / pakietu
            warnings.warn(f"Nie wczytano tokenizera {name} ({e}); liczę tokeny regexem — "
                          f"rozmiary chunków będą przybliżone.")

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Pozycje znakowe kolejnych tokenów."""
        if self._tok is not None:
            return list(self._tok.encode(text, add_special_tokens=False).offsets)
        return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]

    # tekst:   "Klient: nie śpię"
    # tokeny:  [Kl][ient][:][ nie][ ś][pi][ę]
    # offsets: [(0,2),(2,6),(6,7),(7,11),(11,13),(13,15),(15,16)]

    def count(self, text: str) -> int:
        return len(self.offsets(text))


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    return Tokenizer(config.TOKENIZER_NAME)


# --- wspólne -----------------------------------------------------------------
def _unique(seq):
    return list(dict.fromkeys(x for x in seq if x))


def _make_chunk(session: Session, strategy: str, index: int, text: str,
                turns: list[Turn], n_tokens: int) -> Chunk:
    starts = [t.start for t in turns if t.start is not None]
    ends = [t.end for t in turns if t.end is not None]
    return Chunk(
        chunk_id=f"{session.session_id}::{strategy}::{index:04d}",
        session_id=session.session_id,
        chunk_index=index,
        strategy=strategy,
        text=text,
        turn_ids=[t.turn_id for t in turns],
        speaker_ids=_unique(t.speaker_id for t in turns),
        speaker_roles=_unique(t.speaker_role for t in turns),
        start_seconds=min(starts) if starts else None,
        end_seconds=max(ends) if ends else None,
        n_tokens=n_tokens,
    )


# --- 1. stały rozmiar tokenów -------------------------------------------------
def chunk_fixed_tokens(session: Session, strategy: str, size: int, overlap: int = 0) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("overlap musi być w przedziale [0, size)")
    tok = get_tokenizer()

    # sklejamy tury i zapamiętujemy, gdzie w tekście leży każda z nich
    parts, spans, pos = [], [], 0
    for t in session.turns:
        s = render_turn(t)
        parts.append(s)
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1  # +1 za "\n"
    full = "\n".join(parts)

    offs = tok.offsets(full)
    step = size - overlap
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(offs), step)):
        window = offs[start:start + size]
        c0, c1 = window[0][0], window[-1][1]
        text = full[c0:c1].strip()
        turns = [t for t, (a, b) in zip(session.turns, spans) if a < c1 and b > c0]
        chunks.append(_make_chunk(session, strategy, i, text, turns, len(window)))
        if start + size >= len(offs):
            break
    return chunks


# --- 2. okno N tur --------------------------------------------------------------
def chunk_turn_window(session: Session, strategy: str, n_turns: int, overlap_turns: int = 0) -> list[Chunk]:
    if not 0 <= overlap_turns < n_turns:
        raise ValueError("overlap_turns musi być w przedziale [0, n_turns)")
    tok = get_tokenizer()
    step = n_turns - overlap_turns
    turns = session.turns
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(turns), step)):
        window = turns[start:start + n_turns]
        text = "\n".join(render_turn(t) for t in window)
        chunks.append(_make_chunk(session, strategy, i, text, window, tok.count(text)))
        if start + n_turns >= len(turns):
            break
    return chunks


# --- dispatcher --------------------------------------------------------------------
def chunk_session(session: Session, strategy: str) -> list[Chunk]:
    cfg = dict(config.STRATEGIES[strategy])
    kind = cfg.pop("type")
    if kind == "fixed_tokens":
        return chunk_fixed_tokens(session, strategy, **cfg)
    if kind == "turn_window":
        return chunk_turn_window(session, strategy, **cfg)
    raise ValueError(f"Nieznany typ strategii: {kind}")
