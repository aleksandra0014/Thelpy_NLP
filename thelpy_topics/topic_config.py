"""Settings of the topics for Thelpy_NLP (the same values as Topics_v4 and the audio pipeline)."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]

annomi_csv = root / "data" / "AnnoMI-simple.csv"  # only used by load_annomi, not needed here
out = root / "data" / "topic_cache"  # cached sentence embeddings (emb_<name>_<hash>.npy)
annotated = root / "data" / "with_topics"  # session JSON files with "topics" added to every turn

# ---------------------------------------------------------------- session JSON (format of thelpy_rag/loader.py)

# speaker_role in the JSON -> role used by the topic model
roles = {"therapist": "therapist", "client": "client", "patient": "client"}
# when speaker_role is missing: the speaker who asks most and says "you" most is the therapist
therapist_words = ["you", "your", "yours", "yourself", "ty", "ci", "cię", "ciebie", "tobie", "twój", "twoja",
                   "twoje", "pan", "pani", "panu", "panią", "pana"]
min_turn_sec = 0.5  # a turn without a usable end time lasts at least this long

# ---------------------------------------------------------------- topics of one chunk (any number of turns)

# every part of a turn gives its words to its main topic (weight 1) and to its secondary topics (secondary_weight)
chunk_min_share = 0.15  # a secondary topic of a chunk needs at least this share of the chunk's topic weight
chunk_max_secondary = 2  # at most this many secondary topics of a chunk
chunk_max_secondary_subs = 3  # at most this many secondary subtopics of a chunk
share_digits = 3  # shares are rounded in the payload
# payload fields with a keyword index in Qdrant (filtering, e.g. only chunks about alcohol)
indexed_fields = ["topic_main", "topic_all", "subtopic_main", "subtopic_all"]

# multilingual sentence model: English and Polish land in the same vector space
embed_model = "paraphrase-multilingual-mpnet-base-v2"
prototype_langs = ["en", "pl"]

# segmentation
target_words = 25  # a segment is closed when it has at least this many words
max_words = 60  # ... and never gets longer than this
min_words = 4  # a turn with fewer content words is a backchannel ("mm-hmm", "yeah")
words_per_sec = 2.5  # HOPE has no time stamps; also used for the last AnnoMI turn

# context: only what was said BEFORE (own segment + previous informative segments)
weight_self = 0.7
weight_previous = [0.2, 0.1]  # one, two segments back
# generic segments ("a couple of times a week", "it doesn't feel that way") name no topic themselves,
# so they lean more on what was said just before
generic_weight_self = 0.6
generic_weight_previous = [0.25, 0.15]

# sensitive subtopics: allowed only when the segment's OWN text (not the context) is at least this similar to the
# subtopic (calibrated score); talk about health risks, decisions or "I'll think about it" otherwise lands in
# "suicidal thoughts". Tested on 133 AnnoMI sessions: suicidal thoughts in unrelated sessions 38 -> 6,
# self-harm 13 -> 1, OCD 23 -> 1, the authors' topic in the top 3 unchanged (89 of 105)
own_min_score = {"suicidal": 0.55, "self_harm": 0.55, "psychosis": 0.55, "ocd": 0.55, "purging": 0.55,
                 "restriction": 0.55, "binge_eating": 0.55, "sexual_problems": 0.55, "sexual_compulsion": 0.55,
                 "sexual_health": 0.55, "prescription_misuse": 0.55, "pregnancy": 0.55}

# main topic
main_threshold = 0.45  # best similarity needed to get a topic
continue_threshold = 0.40  # the previous topic is kept if it still reaches this (and nothing passed main_threshold)
keep_margin = 0.02  # the previous topic is kept if it is this close to the best one

# frame of the conversation (opening: greeting and small talk; closing: goodbye and the next appointment)
# needs a clear win: high similarity and a margin over the best life topic ...
frame_threshold = 0.66
frame_margin = 0.05
# ... which is easier at the edges: opening in the first, closing in the last part of the session
edge_share = 0.12
edge_threshold = 0.50
edge_margin = 0.02

# short segments whose own text is this far from every life topic are interjections ("if you know what I mean"):
# no topic, no continuation of the previous one
empty_threshold = 0.38
empty_max_words = 10  # ... and only when they have no dictionary word of a life topic

# a segment that is mostly therapy process (scales, decisions, plans) may only continue the previous topic,
# never start a new one; applies when its own text names no life topic (best < main_threshold) ...
process_follows = True
process_lead = 0.08  # ... or when it is this much closer to the process than to the best life topic (raw similarity)
process_open = True  # before the first life topic of the session the usual rules decide (unless clearly process)

# light smoothing of short "islands": A, B, A -> A, A (+B as secondary), A when A also fits the middle segment
island_length = 2  # at most this many segments in a row
island_gap_sec = 30  # the same topic must come back within this time (interjections without a topic are skipped)
island_margin = 0.10  # A may be at most this much below B in the middle segment (0 = no smoothing)
island_tie = 0.025  # ... and when A is at most this much below B, it is always smoothed (practically a draw)
island_keep = 0.60  # a middle segment that clearly names its own topic (B at least this) is never changed

# short turns (backchannels) normally inherit the previous topic, unless they clearly name a topic or are frame talk
short_turn_threshold = 0.55
short_frame_threshold = 0.55
short_keyword_threshold = 0.42  # ... or name a dictionary word whose subtopic reaches this

# calibration: subtopics that attract sentences without a topic get their scores lowered (0 = off)
neutral_correction = 1.0

# secondary topics (max 2)
max_secondary = 2
secondary_threshold = 0.48  # similarity needed for a secondary topic
secondary_margin = 0.06  # ... and it may be at most this much below the main topic
keyword_secondary_threshold = 0.42  # or: a dictionary keyword in the text and its subtopic reaches this

# therapy process layer (compared on raw similarity of the own text, without calibration)
process_threshold = 0.45

keyword_bonus = 0.02  # added to a subtopic when one of its keywords is in the text

# weights for the "sums to 100%" view: main topic 1, every secondary topic 0.5 (then normalised)
secondary_weight = 0.5

window_sec = 60  # windows for the client share line

# thematic blocks (overview plot only, the topics of the segments are not changed)
block_min_sec = 45  # a block is at least this long ...
block_min_share = 0.04  # ... and at least this part of the session
block_topic_share = 0.15  # a topic describes a block when it has at least this share of the block's topic weight

# filler words: a turn with fewer than min_words other words is a backchannel ("mm-hmm", "no tak, jasne"),
# and these words never explain a topic
filler_words_en = ["yeah", "yes", "okay", "ok", "mm", "hmm", "mm-hmm", "um", "uh", "uh-huh", "like", "know", "just",
                   "really", "right", "oh", "well", "so", "gonna", "wanna", "kinda", "sorta", "alright", "yep", "mhm",
                   "dunno", "guess", "suppose", "sort", "kind", "stuff", "thing", "things", "bit", "maybe", "sure",
                   "cool", "no", "nope"]
filler_words_pl = ["tak", "nie", "aha", "aham", "yhm", "ehe", "yyy", "eee", "okej", "dobrze", "jasne", "właśnie",
                   "wiesz", "wie", "jakby", "znaczy", "prostu", "sumie", "niby", "chyba", "może", "trochę", "coś",
                   "tam", "tego", "więc", "dokładnie", "rozumiem", "oczywiście", "spoko", "faktycznie", "rzecz",
                   "rzeczy", "takie", "taki", "jakoś"]
filler_words = filler_words_en + filler_words_pl
