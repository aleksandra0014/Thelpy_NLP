"""Topics of the chunks of a session, without Qdrant and Ollama (to try chunk sizes and thresholds).

    python -m thelpy_topics.preview data/annomi_64.json --n-turns 8
    python -m thelpy_topics.preview data/annomi_64.json --n-turns 4 --overlap-turns 1
    python -m thelpy_topics.preview data/annomi_64.json --size 256          # chunks of 256 tokens

Prints one line per chunk and saves a CSV next to the annotated session (data/with_topics/).
"""
import argparse

import pandas as pd

from thelpy_rag.pipeline import resolve_strategy
from thelpy_topics import integration
from thelpy_topics import topic_config as config
from thelpy_topics.annotate import annotate_files


def chunk_rows(path, strategy):
    """Return one row per chunk: index, turns, time, topics and subtopics with shares."""
    session = integration.load_session_with_topics(path)
    rows = []
    for chunk in integration.chunk_session_with_topics(session, strategy):
        payload = chunk.payload()
        rows.append({
            "chunk": chunk.chunk_index,
            "turns": payload["turn_range"],
            "start": chunk.start_seconds,
            "end": chunk.end_seconds,
            "topic_main": payload.get("topic_main"),
            "topic_secondary": ", ".join(payload.get("topic_secondary", [])),
            "subtopics": ", ".join(payload.get("subtopic_all", [])),
            "topic_shares": payload.get("topic_shares"),
            "none_share": payload.get("topic_none_share"),
            "text": chunk.text,
        })
    return pd.DataFrame(rows)


def print_rows(rows):
    """Print a short line per chunk."""
    for row in rows.itertuples():
        shares = ", ".join(f"{topic} {share:.0%}" for topic, share in row.topic_shares.items())
        main = row.topic_main if isinstance(row.topic_main, str) else "-"
        print(f"[{row.chunk:3d}] {row.turns:12s} {main:12s} | {shares} | sub: {row.subtopics}")


def main():
    """Annotate the session (if needed), cut it like thelpy_rag and show the topics of every chunk."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="session .json")
    parser.add_argument("--n-turns", type=int, default=8, help="turns in a chunk")
    parser.add_argument("--overlap-turns", type=int, default=0)
    parser.add_argument("--size", type=int, help="chunks of this many tokens instead of turns")
    parser.add_argument("--overlap", type=int, default=0, help="token overlap (with --size)")
    parser.add_argument("--force", action="store_true", help="compute the topics again")
    args = parser.parse_args()
    paths = annotate_files([args.file], args.force)
    if not paths:
        raise SystemExit("Not a session file.")
    chunking = "fixed" if args.size else "turns"
    strategy = resolve_strategy(chunking, args.size, args.overlap, args.n_turns, args.overlap_turns)
    rows = chunk_rows(paths[0], strategy)
    print_rows(rows)
    out = config.annotated / f"{paths[0].stem}_chunks_{strategy}.csv"
    rows.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
