"""BM25 jako wektor rzadki w Qdrancie.

Podział pracy:
- po stronie dokumentu liczymy składnik TF z BM25:  tf*(k1+1) / (tf + k1*(1-b+b*len/avg_len))
- IDF dolicza Qdrant (kolekcja ma `modifier=IDF`), więc zawsze jest aktualny dla całej kolekcji
- zapytanie to po prostu waga 1.0 dla każdego unikalnego termu

Indeksy termów to hash crc32 słowa, więc nie trzeba przechowywać słownika.
Tokeny przechodzą przez stemmer Snowball (BM25_STEMMER: english, polish, ... | none),
a dodatkowo można je obciąć do prefiksu (BM25_STEM_PREFIX).
Lista stopwords wybierana jest przez BM25_STOPWORDS (english | polish | none, domyślnie english).
Stopwords są filtrowane przed stemmingiem, więc listy zawierają pełne formy słów.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter

import snowballstemmer
from qdrant_client import models

from . import config

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

POLISH_STOPWORDS = set("""
a aby ale albo ani aż bardzo bez bo być był była było były będzie będą ci co coś czy czyli
dla do gdy gdzie go i ich ile im ja jak jaki jakie jako je jego jej jest jestem jesteś jeszcze
już ją ją kiedy kto która które którego który ku lub ma mam mi mnie mną mu my na nad nam nas
nawet nie nic nich nim no o od on ona one oni ono oraz pan pani po pod podczas przez przy
się są ta tak takie także tam te tego tej ten też to tu tylko tym u w we więc wszystko z za
ze że żeby um mhm yyy eee no tak
""".split())

ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
don doesn didn isn wasn aren weren ve ll re im
um uh mhm hmm yeah oh okay ok like
""".split())

STOPWORDS = {"english": ENGLISH_STOPWORDS, "polish": POLISH_STOPWORDS, "none": set()}


def get_stopwords(name: str = config.BM25_STOPWORDS) -> set[str]:
    try:
        return STOPWORDS[name]
    except KeyError:
        raise ValueError(f"Nieznane BM25_STOPWORDS={name!r}, dostępne: {', '.join(STOPWORDS)}") from None


def get_stemmer(name: str = config.BM25_STEMMER):
    """Stemmer Snowball dla danego języka albo None dla 'none'."""
    if name == "none":
        return None
    if name not in snowballstemmer.algorithms():
        raise ValueError(f"Nieznane BM25_STEMMER={name!r}, dostępne: none, "
                         f"{', '.join(snowballstemmer.algorithms())}")
    return snowballstemmer.stemmer(name)

# tf (term frequency): ile razy słowo występuje we fragmencie.
# IDF (inverse document frequency): jak rzadkie jest słowo w całej kolekcji. Słowo obecne w niewielu fragmentach („bezsenność”) jest ważne, a słowo obecne wszędzie („praca”) mało mówi.
# k1 (1.2) odpowiada za nasycenie. Słowo powtórzone 10 razy nie jest 10 razy ważniejsze niż jedno wystąpienie. Wkład rośnie coraz wolniej i nigdy nie przekroczy k1+1 = 2.2.
# b (0.75) odpowiada za karę za długość. W długim fragmencie łatwiej przypadkiem trafić słowo, więc jego wkład jest pomniejszany. b=0 wyłącza karę, b=1 stosuje ją w pełni.
# avg_len (150) to typowa długość fragmentu. Fragment dłuższy niż średnia jest karany, krótszy nagradzany.


class BM25Encoder:
    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B,
                 avg_len: float = config.BM25_AVG_LEN, stem_prefix: int | None = config.BM25_STEM_PREFIX,
                 stopwords: set[str] | None = None, stemmer: str = config.BM25_STEMMER,
                 min_len: int = 2):
        self.k1, self.b, self.avg_len = k1, b, avg_len
        self.stem_prefix = stem_prefix
        self.stopwords = get_stopwords() if stopwords is None else stopwords
        self.stemmer = get_stemmer(stemmer)
        self.min_len = min_len

    def tokenize(self, text: str) -> list[str]:
        toks = [t for t in TOKEN_RE.findall(text.lower())
                if len(t) >= self.min_len and t not in self.stopwords and not t.isdigit()]
        if self.stemmer:
            toks = self.stemmer.stemWords(toks)
        if self.stem_prefix:
            toks = [t[: self.stem_prefix] for t in toks]
        return toks

    @staticmethod
    def _index(term: str) -> int:
        return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF

    @staticmethod
    def _to_sparse(weights: dict[int, float]) -> models.SparseVector:
        idx = sorted(weights)
        return models.SparseVector(indices=idx, values=[float(weights[i]) for i in idx])

    def encode_document(self, text: str) -> models.SparseVector:
        toks = self.tokenize(text)
        doc_len = len(toks)
        norm = self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
        weights: dict[int, float] = {}
        for term, tf in Counter(toks).items():
            i = self._index(term)
            weights[i] = weights.get(i, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return self._to_sparse(weights)

    def encode_query(self, text: str) -> models.SparseVector:
        return self._to_sparse({self._index(t): 1.0 for t in set(self.tokenize(text))})
