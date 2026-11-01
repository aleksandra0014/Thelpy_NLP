"""Topics in thelpy_rag without changing its files.

enable_topics() swaps a few functions of thelpy_rag for versions that also handle topics:
- loading a session also reads the "topics" of every turn (from a file made by thelpy_topics.annotate),
- every chunk gets the topics of its turns in its payload (topic_main, topic_all, subtopic_all, shares ...),
- optionally: the LLM sees the topics of every fragment, and the search is limited to one topic or subtopic.
Call it before thelpy_rag.pipeline.ingest / query / run.
"""
import json
from pathlib import Path

from qdrant_client import models

from thelpy_rag import llm, pipeline, store
from thelpy_topics import topic_config as config
from thelpy_topics.aggregate import aggregate_topics
from thelpy_topics.annotate import turn_id

original = {
    "load_session": pipeline.load_session,
    "chunk_session": pipeline.chunk_session,
    "build_filter": pipeline.build_filter,
    "format_context": llm.format_context,
}
state = {"topic": None, "turns": {}}


def read_turn_topics(path):
    """Return the session id and {turn_id: topics} of a session file made by thelpy_topics.annotate."""
    session = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = {}
    for index, turn in enumerate(session["turns"]):
        topics[turn_id(turn, index)] = turn.get("topics", [])
    return str(session["session_id"]), topics


def load_session_with_topics(path):
    """Load a session like thelpy_rag does and remember the topics of its turns."""
    session = original["load_session"](path)
    session_id, topics = read_turn_topics(path)
    state["turns"][session_id] = topics
    return session


def chunk_session_with_topics(session, strategy):
    """Cut a session like thelpy_rag does and add the topics of its turns to every chunk."""
    chunks = original["chunk_session"](session, strategy)
    topics = state["turns"].get(session.session_id)
    if not topics:
        return chunks
    for chunk in chunks:
        turn_topics = [topics.get(turn, []) for turn in chunk.turn_ids]
        chunk.extra.update(aggregate_topics(turn_topics))
    return chunks


def topic_note(payload):
    """Return a short line with the topics of a fragment, e.g. "topics: Substances and addictions 62%"."""
    labels = payload.get("topic_labels") or []
    if not labels:
        return ""
    found = payload.get("topic_shares", {})
    topics = [f"{label} {found.get(topic, 0):.0%}" for topic, label in zip(payload["topic_all"], labels)]
    note = "topics: " + ", ".join(topics)
    subs = payload.get("subtopic_labels") or []
    if subs:
        note = note + "; subtopics: " + ", ".join(subs)
    return note


def format_context_with_topics(payloads):
    """Return the fragments for the LLM like thelpy_rag does, with the topics of every fragment in its header."""
    blocks = []
    for payload in payloads:
        header = f"[{payload['session_id']}, {payload.get('turn_range')}, session date: {payload.get('session_at')}]"
        note = topic_note(payload)
        if note:
            header = header + f"\n({note})"
        blocks.append(header + "\n" + payload["text"])
    return "\n\n---\n\n".join(blocks)


def topic_condition(topic):
    """Return a Qdrant condition: the chunk has this topic or subtopic (main or secondary)."""
    by_topic = models.FieldCondition(key="topic_all", match=models.MatchValue(value=topic))
    by_sub = models.FieldCondition(key="subtopic_all", match=models.MatchValue(value=topic))
    return models.Filter(should=[by_topic, by_sub])


def build_filter_with_topics(*args, **kwargs):
    """Build the thelpy_rag filter and, when a topic is chosen, also require that topic."""
    found = original["build_filter"](*args, **kwargs)
    if not state["topic"]:
        return found
    condition = topic_condition(state["topic"])
    if found is None:
        return models.Filter(must=[condition])
    return models.Filter(must=list(found.must or []) + [condition])


def add_payload_indexes():
    """Add the topic fields to the payload indexes made when thelpy_rag creates a collection."""
    known = {field for field, _ in store.PAYLOAD_INDEXES}
    for field in config.indexed_fields:
        if field not in known:
            store.PAYLOAD_INDEXES.append((field, models.PayloadSchemaType.KEYWORD))


def ensure_indexes(client, collection):
    """Create the topic indexes in a collection that already existed before topics were added."""
    for field in config.indexed_fields:
        try:
            client.create_payload_index(collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception as error:  # noqa: BLE001
            print(f"[topics] index {field}: {error}")


def enable_topics(topic=None, prompt=True):
    """Turn on topics in thelpy_rag; topic limits the search to one topic or subtopic, prompt shows topics to the LLM."""
    state["topic"] = topic
    pipeline.load_session = load_session_with_topics
    pipeline.chunk_session = chunk_session_with_topics
    pipeline.build_filter = build_filter_with_topics
    if prompt:
        llm.format_context = format_context_with_topics
    add_payload_indexes()
