"""Topics of every turn of a session JSON (thelpy_rag format), saved as a copy with a "topics" list per turn.

    python -m thelpy_topics.annotate data/annomi_64.json            # -> data/with_topics/annomi_64.json
    python -m thelpy_topics.annotate data/sessions/ --force         # every *.json in a folder, computed again

Topics are found on the whole session (every part of a turn also looks at what was said before), so the
result does not depend on how the session is later cut into chunks.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from thelpy_topics import topic_config as config
from thelpy_topics import topics as tp


def to_seconds(value):
    """Return seconds from a number or from "hh:mm:ss[.ff]" / "mm:ss" (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def turn_id(turn, index):
    """Return the turn id the same way as thelpy_rag/loader.py does."""
    return turn.get("turn_id") or f"T{index + 1:03d}"


def turn_rows(session):
    """Return the non-empty turns of a session as rows: turn, turn_id, speaker_id, role, text, start, end."""
    rows = []
    for index, turn in enumerate(session["turns"]):
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        rows.append({
            "turn": index,
            "turn_id": turn_id(turn, index),
            "speaker_id": str(turn.get("speaker_id", "UNK")),
            "role": config.roles.get(str(turn.get("speaker_role") or "").lower()),
            "text": text,
            "start": to_seconds(turn.get("start")),
            "end": to_seconds(turn.get("end")),
        })
    return pd.DataFrame(rows)


def therapist_score(texts):
    """Return how much a speaker sounds like a therapist: share of questions plus share of "you" words."""
    joined = " ".join(texts)
    sentences = max(len(re.findall(r"[.!?]", joined)), 1)
    questions = joined.count("?") / sentences
    words = tp.words(joined)
    addressed = sum(word in config.therapist_words for word in words) / max(len(words), 1)
    return questions + addressed


def guess_roles(turns):
    """Return a role for every turn: from speaker_role, or guessed per speaker when a role is missing."""
    if turns["role"].notna().all():
        return turns["role"]
    scores = {}
    for speaker, part in turns.groupby("speaker_id"):
        scores[speaker] = therapist_score(part["text"])
    therapist = max(scores, key=scores.get)
    guessed = turns["speaker_id"].map(lambda speaker: "therapist" if speaker == therapist else "client")
    return turns["role"].fillna(guessed)


def fill_starts(turns):
    """Return start times; a missing start is the end of the previous turn (0 for the first one)."""
    starts = []
    previous_end = 0.0
    for row in turns.itertuples():
        start = previous_end if pd.isna(row.start) else row.start
        starts.append(start)
        words = len(row.text.split())
        previous_end = start + words / config.words_per_sec if pd.isna(row.end) else row.end
    return starts


def fill_ends(turns):
    """Return end times: the given end if it is after the start, else the next start or a guess from words."""
    next_start = turns["start"].shift(-1)
    guess = turns["start"] + turns["text"].str.split().str.len() / config.words_per_sec
    usable = turns["end"].notna() & (turns["end"] > turns["start"])
    end = turns["end"].where(usable, next_start)
    end = end.where(end > turns["start"], guess)
    return end.clip(lower=turns["start"] + config.min_turn_sec)


def session_turns(session):
    """Return the turns in the format of the topic model: session, turn, speaker, text, start, end, turn_id."""
    turns = turn_rows(session)
    if turns.empty:
        raise ValueError(f"Session {session.get('session_id')} has no non-empty turns.")
    turns = turns.assign(speaker=guess_roles(turns), session=str(session["session_id"]))
    turns = turns.assign(start=fill_starts(turns))
    return turns.assign(end=fill_ends(turns))


class TopicModel:
    """Sentence model and topic prototypes; loaded once and used for many sessions."""

    def __init__(self):
        self.model = tp.load_model()
        self.protos, _, _ = tp.build_prototypes(self.model)
        self.bias = tp.neutral_bias(self.model, self.protos)

    def score(self, turns, name):
        """Return the topic parts of all turns (the same steps as in Topics_v4)."""
        parts = tp.make_segments(turns)
        emb = tp.encode(self.model, parts["text"], name)
        return tp.score_segments(parts, emb, self.protos, self.bias)


def topics_version():
    """Return a short fingerprint of the dictionary, the settings and the code of the topics (changes with them)."""
    folder = Path(__file__).resolve().parent
    digest = hashlib.md5()
    for name in ["taxonomy.py", "topic_config.py", "topics.py"]:
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()[:10]


def part_entry(row):
    """Return one part of a turn as plain JSON values."""
    return {
        "part": int(row["part"]),
        "words": int(row["n_words"]),
        "start": round(float(row["start"]), 2),
        "end": round(float(row["end"]), 2),
        "main": row["main"],
        "main_sub": row["main_sub"],
        "secondary": list(row["secondary"]),
        "secondary_subs": list(row["secondary_subs"]),
        "process": row["process"],
        "source": row["source"],
    }


def turn_topics(parts):
    """Return {turn index: [parts of the turn]}."""
    found = {}
    for _, row in parts.sort_values(["turn", "part"]).iterrows():
        found.setdefault(int(row["turn"]), []).append(part_entry(row))
    return found


def add_topics(session, model):
    """Return a copy of the session with a "topics" list in every turn (empty turns get an empty list)."""
    turns = session_turns(session)
    found = turn_topics(model.score(turns, f"session_{session['session_id']}"))
    roles = dict(zip(turns["turn"], turns["speaker"]))
    new_turns = []
    for index, turn in enumerate(session["turns"]):
        extra = {"topics": found.get(index, [])}
        if index in roles:
            extra["topic_role"] = roles[index]
        new_turns.append({**turn, **extra})
    return {**session, "turns": new_turns, "topics_model": config.embed_model, "topics_version": topics_version()}


def annotated_path(path):
    """Return where the copy with topics of a session file is saved."""
    return config.annotated / Path(path).name


def is_fresh(path):
    """Tell if the copy with topics exists, is newer than the session file and was made with the current topics."""
    out = annotated_path(path)
    if not out.exists() or out.stat().st_mtime < Path(path).stat().st_mtime:
        return False
    saved = json.loads(out.read_text(encoding="utf-8")).get("topics_version")
    return saved == topics_version()


def annotate_file(path, model=None, force=False):
    """Save a copy of the session file with topics (skipped if it is up to date); return its path."""
    path = Path(path)
    out = annotated_path(path)
    if is_fresh(path) and not force:
        return out
    model = model or TopicModel()
    session = json.loads(path.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(add_topics(session, model), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def session_files(paths):
    """Return the *.json files from files and folders."""
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files = files + sorted(path.glob("*.json"))
        else:
            files.append(path)
    return files


def is_session(path):
    """Tell if a JSON file is a session (has "session_id" and "turns"), e.g. not a benchmark file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "turns" in data and "session_id" in data


def annotate_files(paths, force=False):
    """Annotate every session file; the model loads only when a file needs it. Return the new paths."""
    model = None
    out = []
    for path in session_files(paths):
        if not is_session(path):
            print(f"[topics] skipped, not a session: {path}")
            continue
        needed = force or not is_fresh(path)
        if needed and model is None:
            model = TopicModel()
        out.append(annotate_file(path, model, force))
    return out


def main():
    """Annotate the session files given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="session .json files or folders")
    parser.add_argument("--force", action="store_true", help="compute again even if the copy exists")
    args = parser.parse_args()
    for path in annotate_files(args.paths, args.force):
        print(path)


if __name__ == "__main__":
    main()
