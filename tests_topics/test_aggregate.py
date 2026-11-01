from thelpy_topics.aggregate import aggregate_topics


def part(words, main, sub, secondary=(), secondary_subs=(), process=""):
    return {"words": words, "main": main, "main_sub": sub, "secondary": list(secondary),
            "secondary_subs": list(secondary_subs), "process": process}


def test_main_topic_has_largest_share():
    turns = [[part(30, "substances", "alcohol")], [part(10, "family", "mother")]]
    result = aggregate_topics(turns)
    assert result["topic_main"] == "substances"
    assert result["topic_secondary"] == ["family"]
    assert result["topic_shares"] == {"substances": 0.75, "family": 0.25}
    assert result["subtopic_all"] == ["alcohol", "mother"]


def test_secondary_topic_counts_half_and_small_shares_are_dropped():
    turns = [[part(40, "substances", "alcohol", ["social"], ["party_scene"])], [part(2, "work", "job")]]
    result = aggregate_topics(turns)
    assert result["topic_shares"]["social"] == round(20 / 62, 3)
    assert "work" not in result["topic_secondary"]


def test_parts_of_one_turn_count_separately():
    long_turn = [part(25, "family", "mother"), part(25, "work", "job")]
    result = aggregate_topics([long_turn])
    assert set(result["topic_all"]) == {"family", "work"}


def test_chunk_without_topics():
    result = aggregate_topics([[part(3, "none", "none", process="scaling")], []])
    assert result["topic_main"] is None
    assert result["topic_all"] == []
    assert result["topic_none_share"] == 1.0
    assert result["topic_process_share"] == 1.0
