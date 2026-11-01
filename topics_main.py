"""main.py with topics: the same options as main.py, plus topics of every chunk in Qdrant and in the LLM prompt.

Before ingest every session file gets topics for every turn (saved in data/with_topics/, the original
file is not changed). Then every chunk gets the topics of its turns in its payload.

Examples:
  # window of 8 turns, hybrid, answer from the LLM with topics in the prompt
  python topics_main.py --file data/annomi_64.json --chunking turns --n-turns 8 --query "What drugs does he use?" --ask

  # only chunks about alcohol (a main topic or a subtopic key from thelpy_topics/taxonomy.py)
  python topics_main.py --chunking turns --n-turns 8 --skip-ingest --topic alcohol --query "how much does he drink"
"""
import argparse
import textwrap

from thelpy_rag import config
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.sparse import BM25Encoder
from thelpy_rag.store import get_client
from thelpy_topics.annotate import annotate_files
from thelpy_topics.integration import enable_topics, ensure_indexes, topic_note


def parse_args():
    """Read the options of main.py and the topic options."""
    parser = argparse.ArgumentParser(description="Thelpy RAG with topics: ingest + search")
    group = parser.add_argument_group("data")
    group.add_argument("--file", nargs="+", help="session .json file(s) or folder")
    group.add_argument("--skip-ingest", action="store_true", help="do not embed, only ask")
    group.add_argument("--recreate", action="store_true", help="clear the collection before ingest")
    group = parser.add_argument_group("chunking")
    group.add_argument("--chunking", default="turns", help="'fixed', 'turns' or a name from config.STRATEGIES")
    group.add_argument("--size", type=int, help="fixed: chunk size in tokens")
    group.add_argument("--overlap", type=int, default=0, help="fixed: overlap in tokens")
    group.add_argument("--n-turns", type=int, default=4, help="turns: turns in a chunk")
    group.add_argument("--overlap-turns", type=int, default=0, help="turns: shared turns between windows")
    group = parser.add_argument_group("search")
    group.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    group.add_argument("--fusion", default="client", choices=["client", "server"])
    group.add_argument("-k", type=int, default=5)
    group.add_argument("--query", action="append", default=[], help="question (can be repeated)")
    group.add_argument("--questions", help="text file with one question per line")
    group.add_argument("--session", help="filter: session_id")
    group.add_argument("--client", help="filter: client_id")
    group.add_argument("--neighbors", type=int, default=0, help="add N neighbouring chunks")
    group.add_argument("--ask", action="store_true", help="answer with the LLM (config.LLM_MODEL)")
    group = parser.add_argument_group("topics")
    group.add_argument("--topic", help="search only chunks with this topic or subtopic, e.g. substances, alcohol")
    group.add_argument("--no-topic-prompt", action="store_true", help="do not show topics to the LLM")
    group.add_argument("--force-topics", action="store_true", help="compute the topics of the sessions again")
    return parser.parse_args()


def read_questions(args):
    """Return the questions from --query and --questions."""
    questions = list(args.query)
    if args.questions:
        with open(args.questions, encoding="utf-8") as file:
            questions = questions + [line.strip() for line in file if line.strip()]
    if not questions:
        raise SystemExit("Give --query or --questions")
    return questions


def print_result(result):
    """Print the found fragments with their topics and the LLM answer."""
    print(f"\n=== {result['question']}   [{result['strategy']} | {result['mode']}]")
    for number, hit in enumerate(result["hits"], 1):
        payload = hit.payload
        ranks = f"  ranks={hit.ranks}" if hit.ranks else ""
        print(f"{number}. {hit.score:.4f}  {payload['session_id']}  {payload.get('turn_range')}{ranks}")
        print(f"   ({topic_note(payload) or 'no topic'})")
        print(textwrap.indent(textwrap.shorten(payload["text"], 300), "   "))
    if "answer" in result:
        print(f"\n--- answer ---\n{result['answer']}")


def main():
    """Annotate the sessions with topics, ingest them like main.py and answer the questions."""
    args = parse_args()
    questions = read_questions(args)
    if not args.file and not args.skip_ingest:
        raise SystemExit("Give --file or use --skip-ingest")
    enable_topics(args.topic, prompt=not args.no_topic_prompt)
    from thelpy_rag.pipeline import ingest, query, resolve_strategy

    strategy = resolve_strategy(args.chunking, args.size, args.overlap, args.n_turns, args.overlap_turns)
    client, embedder, bm25 = get_client(), OllamaEmbedder(), BM25Encoder()
    if args.file and not args.skip_ingest:
        files = [str(path) for path in annotate_files(args.file, args.force_topics)]
        ingest(files, strategy, args.recreate, client, embedder, bm25)
        ensure_indexes(client, config.collection_name(strategy))
    for question in questions:
        result = query(question, strategy, args.mode, args.k, args.session, args.client, args.neighbors,
                       args.fusion, args.ask, client, embedder, bm25)
        print_result(result)


if __name__ == "__main__":
    main()
