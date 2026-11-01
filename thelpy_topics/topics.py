"""Topic pipeline: load a transcript, cut it into segments, assign a main topic and up to two
secondary topics (with subtopics), explain the decision and aggregate."""
import hashlib
import re

import numpy as np
import pandas as pd

from thelpy_topics import taxonomy as tax
from thelpy_topics import topic_config as cfg

none_topic = "none"
letters = r"\wąćęłńóśźż'-"


# ---------------------------------------------------------------- data

def to_seconds(text):
    """Convert 'hh:mm:ss' to seconds."""
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def load_annomi(session):
    """Return turns of one AnnoMI session: session, turn, speaker, text, start, end."""
    df = pd.read_csv(cfg.annomi_csv)
    df = df[df["transcript_id"] == session].sort_values("utterance_id")
    turns = pd.DataFrame({
        "session": session,
        "turn": df["utterance_id"].values,
        "speaker": df["interlocutor"].values,
        "text": df["utterance_text"].astype(str).values,
        "start": df["timestamp"].map(to_seconds).astype(float).values,
    })
    # AnnoMI has only start times: a turn ends when the next one starts
    turns["end"] = turns["start"].shift(-1)
    n_words = turns["text"].str.split().str.len()
    last = turns["end"].isna()
    turns.loc[last, "end"] = turns.loc[last, "start"] + n_words[last] / cfg.words_per_sec
    turns["end"] = np.maximum(turns["end"], turns["start"] + 0.5)
    return turns


def annomi_info(session):
    """Topic given by the AnnoMI authors, MI quality and video title."""
    df = pd.read_csv(cfg.annomi_csv)
    row = df[df["transcript_id"] == session].iloc[0]
    return {"topic": row["topic"], "mi_quality": row["mi_quality"], "title": row["video_title"]}


# ---------------------------------------------------------------- segments

fillers = set(cfg.filler_words)


def words(text):
    """Lower-case word tokens."""
    return re.findall(f"[{letters}]+", text.lower())


def is_backchannel(text):
    """True for short turns without content (mm-hmm, yeah, okay, right)."""
    content = [w for w in words(text) if w not in fillers]
    return len(content) < cfg.min_words


def split_sentences(text):
    """Split a turn into sentences at . ! ? and long dashes."""
    parts = re.split(r"(?<=[.!?])\s+|\s+[—–]\s+|(?<=[—–])\s*", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_turn(text):
    """Join sentences of one turn into segments of target_words..max_words words."""
    chunks = []
    current = []
    for sentence in split_sentences(text):
        sentence_words = sentence.split()
        while len(sentence_words) > cfg.max_words:
            room = cfg.max_words - len(current)
            chunks.append(" ".join(current + sentence_words[:room]))
            sentence_words = sentence_words[room:]
            current = []
        if len(current) + len(sentence_words) > cfg.max_words:
            chunks.append(" ".join(current))
            current = []
        current = current + sentence_words
        if len(current) >= cfg.target_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        if chunks and len(current) < cfg.target_words / 2:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


def make_segments(turns):
    """Cut turns into segments; the time of a turn is shared by its segments in proportion to words."""
    rows = []
    for t in turns.itertuples():
        backchannel = is_backchannel(t.text)
        chunks = [t.text] if backchannel else chunk_turn(t.text)
        sizes = np.array([max(len(c.split()), 1) for c in chunks], dtype=float)
        edges = t.start + (t.end - t.start) * np.concatenate([[0], np.cumsum(sizes) / sizes.sum()])
        for i, chunk in enumerate(chunks):
            rows.append({"session": t.session, "turn": t.turn, "part": i, "speaker": t.speaker, "text": chunk,
                         "n_words": int(sizes[i]), "start": edges[i], "end": edges[i + 1],
                         "informative": not backchannel})
    seg = pd.DataFrame(rows)
    seg["duration"] = seg["end"] - seg["start"]
    return seg


# ---------------------------------------------------------------- embeddings

def load_model():
    """Load the multilingual sentence model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.embed_model)


def encode(model, texts, name):
    """Embed texts (normalised); results are cached in outputs/emb_<name>_<hash>.npy."""
    texts = list(texts)
    key = hashlib.md5(("\n".join(texts) + cfg.embed_model).encode()).hexdigest()[:10]
    path = cfg.out / f"emb_{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200)
    cfg.out.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def normalise(x):
    """Scale rows to length 1."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------- dictionary

sub_keys = list(tax.subtopics)
sub_main = np.array([tax.subtopics[k]["main"] for k in sub_keys])


def build_prototypes(model):
    """Prototype of every subtopic (mean of description + examples) and the table of example sentences."""
    pairs = []
    for key, s in tax.subtopics.items():
        for lang in cfg.prototype_langs:
            pairs.append((key, s[f"desc_{lang}"], "opis"))
            for example in s[f"examples_{lang}"]:
                pairs.append((key, example, "przykład"))
    examples = pd.DataFrame(pairs, columns=["sub", "text", "kind"])
    emb = encode(model, examples["text"], "prototypes")
    protos = np.stack([emb[(examples["sub"] == k).values].mean(axis=0) for k in sub_keys])
    return normalise(protos), examples, emb


def neutral_bias(model, protos):
    """How much every subtopic attracts sentences without a topic (vague, filler talk from the dictionary).
    Returned relative to the average subtopic: positive = the subtopic attracts everything and its scores
    are lowered by this amount."""
    texts = tax.neutral_examples_en + tax.neutral_examples_pl
    emb = encode(model, texts, "neutral")
    attraction = (emb @ protos.T).mean(axis=0)
    # only lower the "magnets"; topics that attract less than average are not boosted
    return cfg.neutral_correction * np.clip(attraction - attraction.mean(), 0, None)


def keyword_pattern(keyword):
    """Dictionary keyword -> regex: * = any ending, a space = a phrase. A Polish two-word phrase ending with "się"
    also matches in the other order ("pocięłam się" and "się pocięłam")."""
    parts = [re.escape(p).replace(r"\*", f"[{letters}]*") for p in keyword.lower().split()]
    phrase = r"\s+".join(parts)
    reflexive = len(parts) == 2 and parts[1] == "się"
    if reflexive:
        phrase = f"(?:{phrase}|się\\s+{parts[0]})"
    return re.compile(f"(?<![{letters}])" + phrase + f"(?![{letters}])")


patterns = {k: [keyword_pattern(w) for w in s["keywords_en"] + s["keywords_pl"]] for k, s in tax.subtopics.items()}


def keyword_hits(text, sub):
    """Words of the text that match keywords of one subtopic."""
    low = text.lower()
    hits = []
    for pattern in patterns[sub]:
        hits = hits + pattern.findall(low)
    return sorted(set(hits))


def keyword_matrix(texts):
    """(texts x subtopics) matrix: 1 when the text has a keyword of the subtopic."""
    out = np.zeros((len(texts), len(sub_keys)))
    for i, text in enumerate(texts):
        for j, key in enumerate(sub_keys):
            if keyword_hits(text, key):
                out[i, j] = 1
    return out


# ---------------------------------------------------------------- scoring

def past_context(seg, emb, generic=None):
    """Mix every informative segment with the informative segments said BEFORE it (never after).
    Generic segments (own text names no topic clearly) lean more on what was said just before."""
    ctx = emb.copy()
    idx = list(seg.index[seg["informative"]])
    for pos, i in enumerate(idx):
        own_weight, weights = cfg.weight_self, cfg.weight_previous
        if generic is not None and generic[i]:
            own_weight, weights = cfg.generic_weight_self, cfg.generic_weight_previous
        mixed = own_weight * emb[i]
        weight = own_weight
        for back, w in enumerate(weights, start=1):
            if pos - back >= 0:
                mixed = mixed + w * emb[idx[pos - back]]
                weight = weight + w
        ctx[i] = mixed / weight
    return normalise(ctx)


def topic_scores(sub_sim):
    """Main topic score = best subtopic score inside it. Returns {main: array}."""
    return {m: sub_sim[:, sub_main == m].max(axis=1) for m in tax.main_topics}


def best_sub(sub_score_row, main):
    """Best subtopic inside one main topic."""
    cols = np.where(sub_main == main)[0]
    return sub_keys[cols[sub_score_row[cols].argmax()]]


def allowed_frame(scores, position):
    """Frame topic that makes sense at this point: the opening in the first half of the session, the closing in
    the second half ("Thank you" right after "Welcome" is still the opening)."""
    frame = "opening" if position < 0.5 else "closing"
    return frame


def frame_win(scores, position, named=()):
    """Opening or closing when it clearly wins over the life topics. Easier at the start / end of the session,
    when the text also has a dictionary word of that frame topic (named). Returns the frame topic or None."""
    best = allowed_frame(scores, position)
    best_life = max(tax.mains_of_kind("life"), key=lambda m: scores[m])
    at_edge = position <= cfg.edge_share or position >= 1 - cfg.edge_share
    at_edge = at_edge and best in named
    threshold, margin = (cfg.edge_threshold, cfg.edge_margin) if at_edge else (cfg.frame_threshold, cfg.frame_margin)
    if scores[best] >= threshold and scores[best] - scores[best_life] >= margin:
        return best
    return None


def choose_main(scores, previous, own=None, position=0.5, process="", n_words=99, named=(), process_lead=0.0, started=True):
    """Main topic of one segment from its scores (life topics + frame), its own-text scores and the previous
    main topic. position = where the segment is in the session (0 = start, 1 = end); process_lead = how much
    the own text is closer to the therapy process than to the best life topic; started = a life topic was
    already discussed earlier in the session."""
    frame = frame_win(scores, position, named)
    if frame is not None:
        return frame, "początek/koniec rozmowy: wyraźnie ponad próg"
    best = max(tax.mains_of_kind("life"), key=lambda m: scores[m])
    if own is not None:
        own_best = max(own[m] for m in tax.mains_of_kind("life"))
        has_life_word = any(tax.kind(m) == "life" for m in named)
        if own_best < cfg.empty_threshold and n_words <= cfg.empty_max_words and not has_life_word:
            if previous in tax.mains_of_kind("frame"):
                # "I think that's good" after "Shall we meet again in two weeks?" still belongs to the closing
                return previous, "kontynuacja początku/końca rozmowy"
            return none_topic, "wtrącenie bez treści"
        if cfg.process_follows and process and (own_best < cfg.main_threshold or process_lead >= cfg.process_lead):
            # therapy process (scales, decisions, reflections) is about the topic under discussion:
            # it may continue that topic, but never brings in a new one
            if previous in tax.mains_of_kind("life") and scores[previous] >= cfg.continue_threshold:
                return previous, "proces terapii: kontynuacja tematu"
            if started or not cfg.process_open or process_lead >= cfg.process_lead:
                return none_topic, "proces terapii bez tematu z życia"
            # nothing is under discussion yet: the usual rules below decide
    keep_previous = previous in tax.mains_of_kind("life") and previous != best
    if keep_previous and scores[previous] >= cfg.main_threshold and scores[best] - scores[previous] < cfg.keep_margin:
        return previous, "kontynuacja (prawie równy wynik)"
    if scores[best] >= cfg.main_threshold:
        return best, "podobieństwo ≥ próg"
    if previous in tax.mains_of_kind("life") and scores[previous] >= cfg.continue_threshold:
        return previous, "kontynuacja (wynik ≥ próg kontynuacji)"
    return none_topic, "nic nie przekroczyło progu"


def choose_secondary(scores, main, sub_row, kw_row):
    """Up to max_secondary life topics: close to the main topic, or named by a dictionary keyword whose
    subtopic also fits the text. Returns [(topic, subtopic, reason)]."""
    if main == none_topic:
        return []
    found = []
    top = scores[main]
    for m in tax.mains_of_kind("life"):
        if m == main:
            continue
        if scores[m] >= cfg.secondary_threshold and scores[m] >= top - cfg.secondary_margin:
            found.append((m, best_sub(sub_row, m), "blisko tematu głównego", scores[m]))
            continue
        named = [j for j in np.where((sub_main == m) & (kw_row > 0))[0]
                 if sub_row[j] >= cfg.keyword_secondary_threshold]
        if named:
            j = max(named, key=lambda j: sub_row[j])
            found.append((m, sub_keys[j], "słowo ze słownika", float(sub_row[j])))
    found.sort(key=lambda x: -x[3])
    return [(m, sub, reason) for m, sub, reason, _ in found[: cfg.max_secondary]]


def require_own_score(own_sub, ctx_sub):
    """Sensitive subtopics (cfg.own_min_score) are pushed far down in segments whose own text is less similar to
    them than their threshold, both in the own-text and in the context scores."""
    own_sub, ctx_sub = own_sub.copy(), ctx_sub.copy()
    for key, threshold in cfg.own_min_score.items():
        j = sub_keys.index(key)
        weak = own_sub[:, j] < threshold
        own_sub[weak, j] = own_sub[weak, j] - 1.0
        ctx_sub[weak, j] = ctx_sub[weak, j] - 1.0
    return own_sub, ctx_sub


def score_segments(seg, emb, protos, bias):
    """Assign topics to every segment. Adds columns:
    main, main_sub, main_rule, secondary (list of topics), secondary_subs, secondary_rules,
    process, own_scores / ctx_scores (dict topic -> similarity), source."""
    seg = seg.reset_index(drop=True).copy()
    texts = seg["text"].tolist()
    kw = keyword_matrix(texts)
    raw_own = emb @ protos.T
    own_sub = raw_own - bias + kw * cfg.keyword_bonus
    life = np.isin(sub_main, tax.mains_of_kind("life"))
    generic = own_sub[:, life].max(axis=1) < cfg.main_threshold
    ctx = past_context(seg, emb, generic)
    ctx_sub = ctx @ protos.T - bias + kw * cfg.keyword_bonus
    own_sub, ctx_sub = require_own_score(own_sub, ctx_sub)
    own_main = topic_scores(own_sub)
    ctx_main = topic_scores(ctx_sub)
    columns = {c: [] for c in ["main", "main_sub", "main_rule", "secondary", "secondary_subs", "secondary_rules",
                               "process", "own_scores", "ctx_scores", "source"]}
    previous = none_topic
    previous_row = None
    started = False
    session_start = seg["start"].min()
    session_length = max(seg["end"].max() - session_start, 1.0)
    for i, row in seg.iterrows():
        position = (row["start"] - session_start) / session_length
        own = {m: float(own_main[m][i]) for m in tax.main_topics}
        scores = {m: float(ctx_main[m][i]) for m in tax.main_topics}
        if not row["informative"]:
            # short answers take the topics of the previous informative segment (never of the next one),
            # unless they are clearly the opening or the closing ("Well, welcome", "Thank you, bye")
            best_life = max(tax.mains_of_kind("life"), key=lambda m: own[m])
            if own[best_life] >= cfg.short_turn_threshold:
                # a short turn that clearly names a topic ("drugs, I guess") gets its own topic
                values = [best_life, best_sub(own_sub[i], best_life), "krótka wypowiedź z wyraźnym tematem",
                          [], [], [], ""]
            elif len(short_named := [j for j in np.where((kw[i] > 0) & life)[0]
                                     if own_sub[i, j] >= cfg.short_keyword_threshold]):
                # a short turn with a dictionary word that also fits its meaning ("I know Family Bar, yes")
                j = max(short_named, key=lambda j: own_sub[i, j])
                values = [sub_main[j], sub_keys[j], "krótka wypowiedź ze słowem ze słownika", [], [], [], ""]
            elif own[allowed_frame(own, position)] >= cfg.short_frame_threshold:
                frame = allowed_frame(own, position)
                values = [frame, best_sub(own_sub[i], frame), "krótka wypowiedź: początek/koniec rozmowy",
                          [], [], [], ""]
            elif previous_row is None:
                values = [none_topic, none_topic, "krótka wypowiedź, brak wcześniejszego tematu", [], [], [], ""]
            else:
                values = [previous_row["main"], previous_row["main_sub"], "odziedziczony z poprzedniego fragmentu",
                          previous_row["secondary"], previous_row["secondary_subs"],
                          ["odziedziczony"] * len(previous_row["secondary"]), ""]
            for key, value in zip(["main", "main_sub", "main_rule", "secondary", "secondary_subs",
                                   "secondary_rules", "process"], values):
                columns[key].append(value)
            columns["own_scores"].append(own)
            columns["ctx_scores"].append(scores)
            columns["source"].append("odziedziczony" if values[2].startswith("odziedziczony") else "własny tekst")
            continue
        # therapy process: own text (without calibration) is closer to a process subtopic than to any other topic
        process = ""
        raw = {m: raw_own[i, sub_main == m].max() for m in tax.main_topics}
        others = max(v for m, v in raw.items() if tax.kind(m) != "process")
        process_score = max(raw[m] for m in tax.mains_of_kind("process"))
        if process_score >= cfg.process_threshold and process_score > others:
            process = best_sub(raw_own[i], "process")
        named = set(sub_main[kw[i] > 0])
        life_raw = max(raw[m] for m in tax.mains_of_kind("life"))
        main, rule = choose_main(scores, previous, own, position, process, len(row["text"].split()), named,
                                 process_score - life_raw, started)
        started = started or main in tax.mains_of_kind("life")
        secondary = choose_secondary(scores, main, ctx_sub[i], kw[i])
        main_sub = none_topic if main == none_topic else best_sub(ctx_sub[i], main)
        # where the main decision came from
        if main == none_topic:
            source = "brak tematu"
        elif rule.startswith("kontynuacja"):
            source = "kontynuacja"
        elif max(own, key=lambda m: own[m] if tax.kind(m) != "process" else -1) == main:
            source = "własny tekst"
        else:
            source = "kontekst wcześniejszy"
        values = {"main": main, "main_sub": main_sub, "main_rule": rule,
                  "secondary": [m for m, _, _ in secondary],
                  "secondary_subs": [sub for _, sub, _ in secondary],
                  "secondary_rules": [r for _, _, r in secondary], "process": process,
                  "own_scores": own, "ctx_scores": scores, "source": source}
        for key, value in values.items():
            columns[key].append(value)
        previous = main
        previous_row = values
    for key, value in columns.items():
        seg[key] = value
    if cfg.island_margin > 0:
        seg = smooth_islands(seg, ctx_sub)
    return seg


def smooth_islands(seg, ctx_sub):
    """Light smoothing: a short "island" (1-2 segments) whose main topic differs from the SAME topic before and
    after it (alcohol, X, alcohol) takes the surrounding topic when that topic also fits it almost as well
    (score at least continue_threshold and at most island_margin below its own best). Its own topic is kept
    as a secondary topic, so the information is not lost. Not changed (unless the two topics are practically
    tied, island_tie): segments that name their own topic with a dictionary word ("my mum") or score high
    for it (>= island_keep). The opening / closing and "no topic" are never changed."""
    seg = seg.copy()
    life = tax.mains_of_kind("life")
    # neighbours are the nearest segments with a life topic ("no topic" interjections in between are skipped)
    idx = list(seg.index[seg["informative"] & seg["main"].isin(life)])
    named = keyword_matrix(seg["text"].tolist())
    changed = []
    for pos in range(1, len(idx) - 1):
        for length in range(1, cfg.island_length + 1):
            if pos + length >= len(idx):
                break
            inside = idx[pos: pos + length]
            before, after = idx[pos - 1], idx[pos + length]
            around = seg.at[before, "main"]
            near = seg.at[inside[0], "start"] - seg.at[before, "end"] <= cfg.island_gap_sec \
                and seg.at[after, "start"] - seg.at[inside[-1], "end"] <= cfg.island_gap_sec
            if seg.at[after, "main"] != around or not near:
                continue
            ok = True
            for i in inside:
                own_main = seg.at[i, "main"]
                scores = seg.at[i, "ctx_scores"]
                own_word = named[i, sub_main == own_main].any()  # the text itself names its own topic
                gap = scores[own_main] - scores[around]
                tie = gap <= cfg.island_tie  # practically a draw: always smoothed
                close = gap <= cfg.island_margin and not own_word and scores[own_main] < cfg.island_keep
                ok = ok and own_main != around and scores[around] >= cfg.continue_threshold and (tie or close)
            if ok:
                changed += [(i, around, seg.at[i, "main"], seg.at[before, "main_sub"]) for i in inside]
                break
    for i, around, own_main, around_sub in changed:
        secondary = [own_main] + [m for m in seg.at[i, "secondary"] if m not in (own_main, around)]
        subs = [seg.at[i, "main_sub"]] + [sub for m, sub in zip(seg.at[i, "secondary"], seg.at[i, "secondary_subs"])
                                          if m not in (own_main, around)]
        seg.at[i, "main"] = around
        seg.at[i, "main_sub"] = around_sub  # the subtopic of the surrounding talk
        seg.at[i, "main_rule"] = "wygładzenie: wtrącenie między tym samym tematem"
        seg.at[i, "secondary"] = secondary[: cfg.max_secondary]
        seg.at[i, "secondary_subs"] = subs[: cfg.max_secondary]
        seg.at[i, "secondary_rules"] = ["temat własny przed wygładzeniem"] + ["blisko tematu głównego"] * (cfg.max_secondary - 1)
        seg.at[i, "secondary_rules"] = seg.at[i, "secondary_rules"][: len(seg.at[i, "secondary"])]
        seg.at[i, "source"] = "wygładzenie"
        # short answers right after the changed segment inherited its old topic: update them too
        j = i + 1
        while j < len(seg) and not seg.at[j, "informative"] and seg.at[j, "main_rule"].startswith("odziedziczony"):
            for key in ["main", "main_sub", "secondary", "secondary_subs"]:
                seg.at[j, key] = seg.at[i, key]
            seg.at[j, "secondary_rules"] = ["odziedziczony"] * len(seg.at[i, "secondary"])
            j += 1
    return seg


# ---------------------------------------------------------------- explanation

def add_explanations(seg, emb, examples, example_emb):
    """Keywords found for every assigned topic and the closest dictionary sentence of the main subtopic."""
    seg = seg.copy()
    kw_main = []
    kw_secondary = []
    nearest = []
    for i, row in enumerate(seg.itertuples()):
        if not row.informative or row.main == none_topic:
            kw_main.append([])
            kw_secondary.append([[] for _ in row.secondary])
            nearest.append("")
            continue
        kw_main.append(main_keywords(row.text, row.main))
        kw_secondary.append([main_keywords(row.text, m) for m in row.secondary])
        mask = (examples["sub"] == row.main_sub).values
        sims = example_emb[mask] @ emb[i]
        nearest.append(examples["text"].values[mask][sims.argmax()])
    seg["kw_main"] = kw_main
    seg["kw_secondary"] = kw_secondary
    seg["example"] = nearest
    return seg


def main_keywords(text, main):
    """Dictionary words of any subtopic of one main topic found in the text."""
    hits = []
    for key in np.array(sub_keys)[sub_main == main]:
        hits = hits + keyword_hits(text, key)
    return sorted(set(hits))


def occlusion(model, seg, protos):
    """Word importance (a simple SHAP-like method): how much the similarity of the own text to an assigned
    subtopic drops when one word is removed. Returns, for every segment, {subtopic: [value per word]}."""
    variants = []
    owners = []
    for i, row in enumerate(seg.itertuples()):
        if not row.informative or row.main == none_topic:
            continue
        tokens = row.text.split()
        variants.append(row.text)
        owners.append((i, -1))
        for j in range(len(tokens)):
            variants.append(" ".join(tokens[:j] + tokens[j + 1:]))
            owners.append((i, j))
    emb = encode(model, variants, "occlusion")
    result = [{} for _ in range(len(seg))]
    full = {}
    for (i, j), e in zip(owners, emb):
        row = seg.iloc[i]
        for sub in [row["main_sub"]] + list(row["secondary_subs"]):
            sim = float(e @ protos[sub_keys.index(sub)])
            if j == -1:
                full[(i, sub)] = sim
                result[i][sub] = []
            else:
                result[i][sub].append(full[(i, sub)] - sim)
    return result


def stop_words():
    """Words that never explain a topic: fillers, English and Polish function words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    polish = {"i", "w", "z", "na", "się", "że", "to", "nie", "jest", "jak", "co", "ale", "o", "do", "mi", "mnie",
              "ja", "ty", "on", "ona", "my", "wy", "oni", "ten", "ta", "te", "tak", "już", "jeszcze", "bo", "czy",
              "po", "od", "za", "przez", "tylko", "a", "u", "by", "być", "był", "była", "było", "są", "jestem"}
    return fillers | set(ENGLISH_STOP_WORDS) | polish


def clean_token(token):
    """Lower-case word without punctuation."""
    return re.sub(f"[^{letters}]", "", token.lower()).strip("-'")


def top_words(text, importance, n=4):
    """The n content words with the highest positive importance."""
    stop = stop_words()
    pairs = sorted(zip(text.split(), importance), key=lambda x: -x[1])
    out = []
    for token, value in pairs:
        clean = clean_token(token)
        if value <= 0 or not clean or clean in stop or clean in out:
            continue
        out.append(clean)
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- aggregation

def life_and_other():
    """Topics shown in summaries: life topics, opening and closing, no topic."""
    return tax.mains_of_kind("life") + tax.mains_of_kind("frame") + [none_topic]


def has_item(lists, item):
    """Tell for every row of a column with lists if the list holds the item."""
    return lists.map(lambda items: item in items)


def block_similarity(a, b):
    """Cosine similarity of two topic-weight vectors (-1 when one of them is empty)."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return a @ b / norm if norm > 0 else -1


def presence(seg):
    """For every topic: share of session time as main topic and as secondary topic (can sum above 100%)."""
    total = seg["duration"].sum()
    rows = []
    for m in life_and_other():
        as_main = seg.loc[seg["main"] == m, "duration"].sum() / total
        as_secondary = seg.loc[has_item(seg["secondary"], m), "duration"].sum() / total
        rows.append({"main": m, "as_main": as_main, "as_secondary": as_secondary, "present": as_main + as_secondary})
    return pd.DataFrame(rows)


def subtopic_presence(seg):
    """Share of session time when a subtopic is present (as main or secondary subtopic)."""
    total = seg["duration"].sum()
    rows = []
    for key in sub_keys:
        as_main = seg.loc[seg["main_sub"] == key, "duration"].sum() / total
        as_secondary = seg.loc[has_item(seg["secondary_subs"], key), "duration"].sum() / total
        if as_main + as_secondary > 0:
            rows.append({"sub": key, "main": tax.subtopics[key]["main"], "as_main": as_main,
                         "as_secondary": as_secondary, "present": as_main + as_secondary})
    return pd.DataFrame(rows).sort_values("present", ascending=False)


def speaker_time(seg, level="main"):
    """Minutes said by therapist and client while a topic (or subtopic) is present, as main or secondary."""
    rows = []
    keys = life_and_other() if level == "main" else sub_keys
    for key in keys:
        if level == "main":
            present = (seg["main"] == key) | has_item(seg["secondary"], key)
        else:
            present = (seg["main_sub"] == key) | has_item(seg["secondary_subs"], key)
        part = seg[present]
        if part.empty:
            continue
        therapist = part.loc[part["speaker"] == "therapist", "duration"].sum() / 60
        client = part.loc[part["speaker"] == "client", "duration"].sum() / 60
        rows.append({"key": key, "therapist": therapist, "client": client,
                     "client_share": client / (therapist + client)})
    return pd.DataFrame(rows)


def client_share_windows(seg, size=None):
    """Share of client words in fixed time windows."""
    size = size or cfg.window_sec
    rows = []
    rate = seg["n_words"] / seg["duration"].clip(lower=0.1)
    for w0 in np.arange(seg["start"].min(), seg["end"].max(), size):
        w1 = w0 + size
        overlap = (np.minimum(seg["end"], w1) - np.maximum(seg["start"], w0)).clip(lower=0)
        spoken = overlap * rate
        if spoken.sum() == 0:
            continue
        client = spoken[seg["speaker"] == "client"].sum() / spoken.sum()
        rows.append({"start": w0, "end": w1, "client_words": client})
    return pd.DataFrame(rows)


def turn_table(seg):
    """One row per original turn (utterance_id): topics of its longest segment, for comparison with other methods."""
    rows = []
    for turn, part in seg.groupby("turn"):
        best = part.sort_values("n_words", ascending=False).iloc[0]
        rows.append({"utterance_id": turn, "speaker": best["speaker"], "start": part["start"].min(),
                     "text": " ".join(part["text"]), "main": best["main"], "main_sub": best["main_sub"],
                     "secondary": ", ".join(best["secondary"]), "secondary_subs": ", ".join(best["secondary_subs"]),
                     "process": best["process"], "source": best["source"]})
    return pd.DataFrame(rows)


def add_word_importance(model, seg, protos):
    """Columns words_raw ({subtopic: importance per word}) and words ({subtopic: top words})."""
    seg = seg.copy()
    raw = occlusion(model, seg, protos)
    seg["words_raw"] = raw
    seg["words"] = [{sub: top_words(text, values) for sub, values in item.items()}
                    for text, item in zip(seg["text"], raw)]
    return seg


# ---------------------------------------------------------------- main topics and blocks


def main_topics_table(seg, top=3, min_present=0.10):
    """Main topics of the session: life topics ranked by how long they were present (main + secondary),
    at most `top`, and only those present at least `min_present` of the time."""
    pres = presence(seg)
    pres = pres[pres["main"].isin(tax.mains_of_kind("life")) & (pres["present"] >= min_present)]
    pres = pres.sort_values("present", ascending=False).head(top)
    subs = subtopic_presence(seg)
    talk = speaker_time(seg, "main").set_index("key")
    rows = []
    for rank, row in enumerate(pres.itertuples(), start=1):
        best = subs[subs["main"] == row.main].head(2)
        rows.append({"miejsce": rank, "temat": tax.main_topics[row.main]["label_pl"],
                     "obecny łącznie": row.present, "jako główny": row.as_main, "jako poboczny": row.as_secondary,
                     "najważniejsze podtematy": ", ".join(tax.subtopics[s]["label_pl"] for s in best["sub"]),
                     "mówi głównie": "pacjent" if talk.loc[row.main, "client_share"] >= 0.5 else "terapeuta"})
    return pd.DataFrame(rows)


def topic_blocks(seg):
    """Bigger thematic blocks of the session, for the overview plot only (the topics of the segments stay as
    they are). Every segment gives weight 1 to its main topic and secondary_weight to each secondary topic
    (topics, not subtopics). Then:
    1. neighbouring stretches with the same set of leading topics are joined,
    2. the shortest stretch below block_min_sec (or block_min_share of the session) is joined with the more
       similar neighbour, again and again, so short side topics disappear into the block around them,
    3. a block is described by the topics with at least block_topic_share of its weight (max 3).
    The opening and the closing at the very start / end are kept as their own blocks."""
    shown = mains_of_kind_any(["life", "frame"])
    total = seg["end"].max() - seg["start"].min()
    min_dur = max(cfg.block_min_sec, cfg.block_min_share * total)
    blocks = []
    for row in seg.sort_values("start").itertuples():
        vec = np.zeros(len(shown))
        if row.main in shown:
            vec[shown.index(row.main)] += row.duration
        for m in row.secondary:
            if m in shown:
                vec[shown.index(m)] += cfg.secondary_weight * row.duration
        blocks.append({"start": row.start, "end": row.end, "vec": vec})

    def leading(block):
        weight = block["vec"].sum()
        if weight == 0:
            return ()
        share = block["vec"] / weight
        order = [i for i in np.argsort(-share) if share[i] >= cfg.block_topic_share][:3]
        return tuple(sorted(shown[i] for i in order))

    def join(a, b):
        return {"start": a["start"], "end": b["end"], "vec": a["vec"] + b["vec"]}

    def is_frame_edge(pos):
        names = leading(blocks[pos])
        return bool(names) and all(tax.kind(m) == "frame" for m in names) and pos in (0, len(blocks) - 1)

    def join_same():
        out = [blocks[0]]
        for block in blocks[1:]:
            if leading(block) == leading(out[-1]):
                out[-1] = join(out[-1], block)
            else:
                out.append(block)
        return out

    blocks = join_same()
    while len(blocks) > 1:
        short = [(b["end"] - b["start"], pos) for pos, b in enumerate(blocks)
                 if b["end"] - b["start"] < min_dur and not is_frame_edge(pos)]
        if not short:
            break
        _, pos = min(short)
        neighbours = [p for p in (pos - 1, pos + 1) if 0 <= p < len(blocks) and not is_frame_edge(p)]
        if not neighbours:
            neighbours = [p for p in (pos - 1, pos + 1) if 0 <= p < len(blocks)]

        similarities = [block_similarity(blocks[pos]["vec"], blocks[p]["vec"]) for p in neighbours]
        other = neighbours[int(np.argmax(similarities))]
        first, second = sorted([pos, other])
        blocks[first:second + 1] = [join(blocks[first], blocks[second])]
        blocks = join_same()

    rows = []
    for number, block in enumerate(blocks, start=1):
        weight = block["vec"].sum()
        names = leading(block)
        shares = {m: block["vec"][shown.index(m)] / weight for m in names} if weight else {}
        names = sorted(names, key=lambda m: -shares[m])
        rows.append({"block": number, "start": block["start"], "end": block["end"],
                     "topics": names if names else [none_topic],
                     "shares": [shares[m] for m in names] if names else [1.0]})
    return pd.DataFrame(rows)


def mains_of_kind_any(kinds):
    """Main topics of the given kinds, in dictionary order."""
    return [m for m in tax.main_topics if tax.kind(m) in kinds]
