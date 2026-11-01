"""Topics of a chunk from the topics of its turns; works for any chunk size (4, 8, 20 turns, tokens, overlap).

Every part of a turn gives its words to its main topic (weight 1) and to each secondary topic
(weight secondary_weight). The chunk gets the topic with the largest share as main topic and up to
chunk_max_secondary further topics with at least chunk_min_share as secondary topics; the same for subtopics.
"""
from thelpy_topics import taxonomy as tax
from thelpy_topics import topic_config as config

none_topic = "none"


def add_weight(weights, key, value):
    """Return the weights with value added to key."""
    return {**weights, key: weights.get(key, 0.0) + value}


def part_weights(part):
    """Return [(topic, subtopic, weight)] of one part of a turn; parts without a topic give nothing."""
    if part["main"] == none_topic:
        return []
    words = part["words"]
    found = [(part["main"], part["main_sub"], float(words))]
    for topic, sub in zip(part["secondary"], part["secondary_subs"]):
        found.append((topic, sub, words * config.secondary_weight))
    return found


def collect(turn_topics):
    """Return topic weights, subtopic weights and word counts (all, without topic, therapy process)."""
    topics, subs = {}, {}
    counts = {"all": 0, "none": 0, "process": 0}
    for parts in turn_topics:
        for part in parts:
            counts = add_weight(counts, "all", part["words"])
            if part["main"] == none_topic:
                counts = add_weight(counts, "none", part["words"])
            if part["process"]:
                counts = add_weight(counts, "process", part["words"])
            for topic, sub, weight in part_weights(part):
                topics = add_weight(topics, topic, weight)
                subs = add_weight(subs, sub, weight)
    return topics, subs, counts


def shares(weights):
    """Return the weights as shares that sum to 1, largest first."""
    total = sum(weights.values())
    if total == 0:
        return {}
    ordered = sorted(weights.items(), key=lambda item: -item[1])
    return {key: round(value / total, config.share_digits) for key, value in ordered}


def leading(found, max_secondary):
    """Return the main key and the secondary keys (share at least chunk_min_share) of the shares."""
    keys = list(found)
    if not keys:
        return None, []
    secondary = [key for key in keys[1:] if found[key] >= config.chunk_min_share]
    return keys[0], secondary[:max_secondary]


def share_of(count, counts):
    """Return count as a share of all words of the chunk."""
    return round(count / max(counts["all"], 1), config.share_digits)


def topic_label(topic):
    """Return the English label of a main topic (None stays None)."""
    if topic is None:
        return None
    return tax.main_topics[topic]["label_en"]


def sub_label(sub):
    """Return the English label of a subtopic (None stays None)."""
    if sub is None:
        return None
    return tax.subtopics[sub]["label_en"]


def aggregate_topics(turn_topics):
    """Return the topic fields of one chunk from the "topics" lists of its turns (ready for the Qdrant payload)."""
    topics, subs, counts = collect(turn_topics)
    topic_shares, sub_shares = shares(topics), shares(subs)
    main, secondary = leading(topic_shares, config.chunk_max_secondary)
    main_sub, secondary_subs = leading(sub_shares, config.chunk_max_secondary_subs)
    all_topics = [main] + secondary if main else []
    all_subs = [main_sub] + secondary_subs if main_sub else []
    return {
        "topic_main": main,
        "topic_main_label": topic_label(main),
        "topic_secondary": secondary,
        "topic_all": all_topics,
        "topic_labels": [topic_label(topic) for topic in all_topics],
        "topic_shares": topic_shares,
        "subtopic_main": main_sub,
        "subtopic_secondary": secondary_subs,
        "subtopic_all": all_subs,
        "subtopic_labels": [sub_label(sub) for sub in all_subs],
        "subtopic_shares": sub_shares,
        "topic_none_share": share_of(counts["none"], counts),
        "topic_process_share": share_of(counts["process"], counts),
    }
