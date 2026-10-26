"""Answer generation from retrieved fragments (Ollama /api/chat)."""
from __future__ import annotations

import warnings

import requests

from . import config
from .chunking import get_tokenizer

SYSTEM_PROMPT = (
    "You are an assistant analysing psychotherapy session transcripts. "
    "Answer only on the basis of the provided fragments. Back every statement with a "
    "reference in the format [session, turns]. If the fragments do not contain the answer, say so explicitly."
)

# chat template overhead (role tags, special tokens) not covered by the plain token count
TEMPLATE_OVERHEAD = 64


class PromptTooLongError(ValueError):
    """The prompt does not fit into num_ctx — Ollama would silently truncate it."""


def format_context(payloads: list[dict]) -> str:
    blocks = []
    for p in payloads:
        blocks.append(f"[{p['session_id']}, {p.get('turn_range')}, session date: {p.get('session_at')}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, payloads: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fragments:\n\n{format_context(payloads)}\n\nQuestion: {question}"},
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


def answer(question: str, payloads: list[dict], model: str = config.LLM_MODEL,
           url: str = config.OLLAMA_URL, temperature: float = config.LLM_TEMPERATURE,
           num_ctx: int = config.LLM_NUM_CTX) -> str:
    messages = build_messages(question, payloads)
    check_prompt_fits(messages, num_ctx)

    r = requests.post(f"{url.rstrip('/')}/api/chat", timeout=600, json={
        "model": model,
        "stream": False,
        "think": False,   # qwen3: disables reasoning mode; remove to enable it
        "options": {"temperature": temperature, "num_ctx": num_ctx},
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
        warnings.warn(f"Answer hit the context/length limit (num_ctx={num_ctx}) and is cut off.")
    return data["message"]["content"]
