"""Answer generation from retrieved fragments (Ollama /api/chat)."""
from __future__ import annotations

import warnings

import requests

from . import config
from .chunking import get_tokenizer

SYSTEM_PROMPT = (
    "You are an assistant analysing psychotherapy session transcripts. "
    "Answer only on the basis of the provided fragments. Back every statement with a "
    "reference in the format [session, turns]. If the fragments do not contain the answer, say so explicitly. "
    "Retrieved fragments show their retrieval rank (1 = judged most relevant to the question) and score; "
    "fragments marked as neighbouring context were added only to show what surrounds a retrieved fragment. "
    "Treat rank as a hint, not as proof: decide by what the fragments actually say."
)

# kody języków -> nazwy dla promptu; inne wartości (np. "Polish", "polski") trafiają do promptu bez zmian
LANGUAGES = {"pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
             "it": "Italian", "uk": "Ukrainian", "ru": "Russian", "cs": "Czech"}


def language_name(language: str | None) -> str | None:
    """"pl" -> "Polish"; pełna nazwa bez zmian; None / pusty -> None (bez instrukcji języka)."""
    if not language or not language.strip():
        return None
    language = language.strip()
    return LANGUAGES.get(language.lower(), language)


def language_note(language: str | None) -> str:
    """Dopisek do promptu systemowego: w jakim języku pisać odpowiedź ("" = bez wymuszania)."""
    name = language_name(language)
    if not name:
        return ""
    return (f" Always write your answer in {name}, regardless of the language of the transcript or the question; "
            f"quote the transcript in its original language only when citing it verbatim.")


def system_prompt(base: str, language: str | None = None) -> str:
    return base + language_note(language)


# chat template overhead (role tags, special tokens) not covered by the plain token count
TEMPLATE_OVERHEAD = 64


class PromptTooLongError(ValueError):
    """The prompt does not fit into num_ctx — Ollama would silently truncate it."""


def fragment_header(p: dict) -> str:
    """[session, turns, date] + linia z rankingiem, jeśli payload przeszedł przez search.build_context."""
    header = f"[{p['session_id']}, {p.get('turn_range')}, session date: {p.get('session_at')}]"
    if "retrieval_rank" in p:
        header += (f"\n(retrieval rank: {p['retrieval_rank']}/{p['retrieval_k']}, "
                   f"score: {p['retrieval_score']:.4f})")
    elif p.get("retrieval_neighbor"):
        header += "\n(neighbouring context, not retrieved)"
    return header


def format_context(payloads: list[dict]) -> str:
    return "\n\n---\n\n".join(f"{fragment_header(p)}\n{p['text']}" for p in payloads)


def build_messages(question: str, payloads: list[dict], formatter=None, language: str | None = None) -> list[dict]:
    """formatter: optional payloads -> context text (default: format_context, e.g. with topics).
    language: język odpowiedzi ("pl", "en", "Polish", ...); None = bez wymuszania (zwykle język pytania)."""
    context = (formatter or format_context)(payloads)
    return [
        {"role": "system", "content": system_prompt(SYSTEM_PROMPT, language)},
        {"role": "user", "content": f"Fragments:\n\n{context}\n\nQuestion: {question}"},
    ]


def count_prompt_tokens(messages: list[dict]) -> int:
    tok = get_tokenizer()
    return sum(tok.count(m["content"]) for m in messages) + TEMPLATE_OVERHEAD


def check_prompt_fits(messages: list[dict], num_ctx: int, reserve: int = config.LLM_ANSWER_RESERVE) -> int:
    """Raises PromptTooLongError if prompt + answer reserve exceeds num_ctx; returns the token estimate."""
    n = count_prompt_tokens(messages)
    if n + reserve > num_ctx:
        raise PromptTooLongError(
            f"Prompt ~{n} tokens + {reserve} reserved for the answer > num_ctx={num_ctx}. "
            f"Reduce k / neighbors or increase LLM_NUM_CTX.")
    return n


def chat(messages: list[dict], model: str = config.LLM_MODEL, url: str = config.OLLAMA_URL,
         temperature: float = config.LLM_TEMPERATURE, num_ctx: int = config.LLM_NUM_CTX,
         num_predict: int | None = None, think: bool = False) -> dict:
    """Single Ollama /api/chat call -> {answer, thinking, prompt_tokens, answer_tokens}. No prompt-size check.
    think: tryb rozumowania (qwen3); tokeny myślenia wliczają się do answer_tokens i do num_predict."""
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict:
        options["num_predict"] = num_predict
    r = requests.post(f"{url.rstrip('/')}/api/chat", timeout=600, json={
        "model": model,
        "stream": False,
        "think": think,   # qwen3: tryb rozumowania (domyślnie wyłączony)
        "options": options,
        "messages": messages,
    })
    r.raise_for_status()
    data = r.json()

    # post-check on Ollama's own count: a truncated prompt fills the whole window
    prompt_tokens = data.get("prompt_eval_count") or 0
    if prompt_tokens >= num_ctx:
        warnings.warn(f"Ollama evaluated {prompt_tokens} prompt tokens = num_ctx={num_ctx}; "
                      f"the prompt was probably truncated.")
    if data.get("done_reason") == "length":
        warnings.warn(f"Answer hit the length limit (num_ctx={num_ctx}, num_predict={num_predict}) "
                      f"and is cut off.")
    return {
        "answer": data["message"]["content"],
        "thinking": data["message"].get("thinking") or None,   # tylko przy think=True
        "prompt_tokens": prompt_tokens,              # Ollama: whole prompt (system + fragments + question)
        "answer_tokens": data.get("eval_count") or 0,
        "truncated": data.get("done_reason") == "length",
    }


def generate(question: str, payloads: list[dict], model: str = config.LLM_MODEL,
             url: str = config.OLLAMA_URL, temperature: float = config.LLM_TEMPERATURE,
             num_ctx: int = config.LLM_NUM_CTX, formatter=None, language: str | None = None,
             think: bool = False) -> dict:
    """Like answer(), but also returns the messages and Ollama's token counts.
    think: włącza rozumowanie qwena; rezerwa na odpowiedź rośnie o LLM_THINK_RESERVE."""
    messages = build_messages(question, payloads, formatter, language)
    check_prompt_fits(messages, num_ctx, reserve=config.LLM_ANSWER_RESERVE + (config.LLM_THINK_RESERVE if think else 0))
    return {"model": model, "messages": messages, "think": think,
            **chat(messages, model=model, url=url, temperature=temperature, num_ctx=num_ctx, think=think)}


def answer(question: str, payloads: list[dict], **kw) -> str:
    return generate(question, payloads, **kw)["answer"]
