from thelpy_topics import topics as tp


def test_polish_reflexive_phrase_matches_both_orders():
    assert tp.keyword_hits("Znowu się pocięłam.", "self_harm") == ["się pocięłam"]
    assert tp.keyword_hits("Myślałem, że się zabiję.", "suicidal") == ["się zabiję"]


def test_ambiguous_words_do_not_match():
    assert tp.keyword_hits("I still think it's important.", "sexual_health") == []
    assert tp.keyword_hits("I want to go to medical school.", "medication") == []
    assert tp.keyword_hits("Mam bliznę po operacji kolana.", "self_harm") == []
    assert tp.keyword_hits("To mi bardzo ciąży.", "pregnancy") == []


def test_sensitive_subtopic_needs_its_own_text():
    import numpy as np

    j = tp.sub_keys.index("suicidal")
    own = np.zeros((2, len(tp.sub_keys)))
    own[0, j], own[1, j] = 0.70, 0.40
    new_own, new_ctx = tp.require_own_score(own, own.copy())
    assert new_own[0, j] == 0.70
    assert new_ctx[1, j] < 0
