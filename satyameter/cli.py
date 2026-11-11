"""Command line entry point.

    python -m satyameter.cli fetch            # download the Wikipedia corpus
    python -m satyameter.cli build-index      # chunk + embed + save FAISS index
    python -m satyameter.cli search "query"   # retrieval only, no LLM
    python -m satyameter.cli check-llm        # test the Groq key / list models
    python -m satyameter.cli ask "question"   # full retrieve -> generate pipeline
    python -m satyameter.cli verify "question"      # ask, then fact-check the answer claim by claim
    python -m satyameter.cli check-claim "claim"    # fact-check one statement against the knowledge base
    python -m satyameter.cli eval-run         # generate claims to hand-label (data/eval/)
    python -m satyameter.cli eval-score       # compute metrics from the labelled CSV
    streamlit run app/streamlit_app.py        # the dashboard
"""

from __future__ import annotations

import argparse
import json
import sys

from satyameter.config import get_settings


def _embedder(settings):
    from satyameter.retrieval.index import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.embed_model)


def cmd_fetch(args) -> int:
    from satyameter.ingestion.wikipedia import download_corpus, read_titles

    s = get_settings()
    titles = read_titles(s.seed_titles)
    print(f"Fetching up to {args.limit or len(titles)} of {len(titles)} articles...")
    ok, failed = download_corpus(titles, s.raw_dir, limit=args.limit)
    print(f"Done: {ok} articles available, {len(failed)} failed.")
    if failed:
        print("Failed titles:", ", ".join(failed))
    return 0 if ok else 1


def cmd_build_index(args) -> int:
    from satyameter.ingestion.chunker import chunk_article
    from satyameter.ingestion.wikipedia import slugify
    from satyameter.retrieval.index import VectorIndex

    s = get_settings()
    files = sorted(s.raw_dir.glob("*.json"))
    if not files:
        print("No articles found. Run: python -m satyameter.cli fetch")
        return 1
    chunks = []
    for path in files:
        article = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(
            chunk_article(
                article["title"],
                article["url"],
                article["text"],
                slug=slugify(article["title"]),
                max_words=s.chunk_words,
                overlap_words=s.chunk_overlap,
            )
        )
    print(f"{len(files)} articles -> {len(chunks)} chunks. Embedding (first run downloads the model)...")
    index = VectorIndex.build(chunks, _embedder(s))
    index.save(s.index_dir, embed_model=s.embed_model)
    print(f"Index saved to {s.index_dir}")
    return 0


def cmd_search(args) -> int:
    from satyameter.retrieval.index import VectorIndex

    s = get_settings()
    index = VectorIndex.load(s.index_dir, _embedder(s))
    for hit in index.search(args.query, k=args.k):
        c = hit.chunk
        print(f"\n#{hit.rank}  score={hit.score:.3f}  {c.doc_title} - {c.section}")
        print(f"    {c.text[:300]}{'...' if len(c.text) > 300 else ''}")
    return 0


def cmd_check_llm(args) -> int:
    import time

    from satyameter.llm import get_llm

    s = get_settings()
    llm = get_llm(s, use_cache=False)
    print(f"Backend/model: {llm.name}")
    start = time.perf_counter()
    reply = llm.generate("You are a helpful assistant.", "Reply with the single word: ready")
    print(f"Reply: {reply!r}  ({time.perf_counter() - start:.1f}s)")
    return 0 if reply else 1


def cmd_ask(args) -> int:
    import time

    from satyameter.generation.rag import answer_question
    from satyameter.llm import get_llm
    from satyameter.retrieval.index import VectorIndex

    s = get_settings()
    index = VectorIndex.load(s.index_dir, _embedder(s))
    llm = get_llm(s)
    start = time.perf_counter()
    result = answer_question(args.question, index, llm, k=args.k)
    elapsed = time.perf_counter() - start
    print(f"\nQ: {result.question}")
    if result.rewritten_query:
        print(f"Retrieval query: {result.rewritten_query}")
    print(f"\nA: {result.answer}\n\nSources:")
    for number, hit in enumerate(result.hits, start=1):
        print(f"  [{number}] {hit.chunk.doc_title} - {hit.chunk.section}  (score {hit.score:.3f})")
        print(f"      {hit.chunk.url}")
    print(f"\n({result.model}, {elapsed:.1f}s)")
    return 0


LABELS = {"SUPPORTED": "[SUPPORTED]   ", "CONTRADICTED": "[CONTRADICTED]", "NOT_FOUND": "[NOT FOUND]   "}


def _print_evidence(verdict) -> None:
    best = verdict.best
    if best is None:
        print("      (no evidence retrieved)")
        return
    c, sc = best.chunk, best.scores
    print(f"      evidence: {c.doc_title} - {c.section}  ({c.url})")
    print(f"      NLI: entail={sc.entailment:.2f} contradict={sc.contradiction:.2f} neutral={sc.neutral:.2f}")
    print(f"      \"{c.text[:220]}{'...' if len(c.text) > 220 else ''}\"")


def cmd_verify(args) -> int:
    from satyameter.generation.rag import answer_question
    from satyameter.llm import get_llm
    from satyameter.retrieval.index import VectorIndex
    from satyameter.verification.nli import NLIModel
    from satyameter.verification.verifier import verify_answer

    s = get_settings()
    index = VectorIndex.load(s.index_dir, _embedder(s))
    llm = get_llm(s)
    nli = NLIModel(s.nli_model, max_length=s.nli_max_length, cpu_threads=s.cpu_threads)
    rag = answer_question(args.question, index, llm, k=s.top_k)
    print(f"\nQ: {rag.question}\n\nA: {rag.answer}\n")
    result = verify_answer(rag, index, llm, nli, claim_k=s.claim_k, max_evidence=s.max_evidence)
    if rag.rewritten_query:
        print(f"Retrieval query: {rag.rewritten_query}\n")
    if not result.verdicts:
        print("No factual claims to verify.")
        return 0
    print("Claim-by-claim verification:")
    for verdict in result.verdicts:
        print(f"\n  {LABELS[verdict.status.value]} {verdict.claim}")
        _print_evidence(verdict)
    print(
        f"\nFaithfulness: {result.faithfulness:.0%}   "
        f"Contradicted: {result.contradiction_rate:.0%}   "
        f"Coverage: {result.coverage:.0%}   ({result.latency_s:.1f}s)"
    )
    return 0


def cmd_check_claim(args) -> int:
    from satyameter.retrieval.index import VectorIndex
    from satyameter.verification.nli import NLIModel
    from satyameter.verification.verifier import verify_claim

    s = get_settings()
    index = VectorIndex.load(s.index_dir, _embedder(s))
    verdict = verify_claim(
        args.claim, index, NLIModel(s.nli_model, max_length=s.nli_max_length, cpu_threads=s.cpu_threads),
        claim_k=args.k, max_evidence=s.max_evidence
    )
    print(f"\n{LABELS[verdict.status.value]} {verdict.claim}")
    _print_evidence(verdict)
    print(f"\nChecked {len(verdict.evidence)} evidence chunks:")
    for e in verdict.evidence:
        print(
            f"  entail={e.scores.entailment:.2f} contradict={e.scores.contradiction:.2f}  "
            f"{e.chunk.doc_title} - {e.chunk.section}"
        )
    return 0


def cmd_eval_run(args) -> int:
    from satyameter.evaluation.run import read_questions, run_eval, write_rows
    from satyameter.pipeline import Satyameter

    s = get_settings()
    out = s.eval_dir / "claims_to_label.csv"
    if out.exists() and not args.force:
        print(f"{out} already exists (it may hold your labels). Use --force to overwrite.")
        return 1
    questions = read_questions(s.eval_dir / "questions.csv", limit=args.limit)
    rows = run_eval(Satyameter(s), questions)
    write_rows(out, rows)
    print(f"\nWrote {len(rows)} claims to {out}")
    print("Open it in Excel, fill the human_ok column (1 = claim is correct, 0 = wrong or made up), save as CSV.")
    return 0


def cmd_eval_score(args) -> int:
    from satyameter.evaluation.score import format_report, load_rows, summarize_eval

    s = get_settings()
    path = s.eval_dir / "claims_to_label.csv"
    if not path.exists():
        print(f"No file at {path}. Run: python -m satyameter.cli eval-run")
        return 1
    summary = summarize_eval(load_rows(path))
    if summary["n_labelled"] == 0:
        print("No labelled rows yet. Fill the human_ok column with 1 or 0 first.")
        return 1
    report = format_report(summary)
    print(report)
    (s.eval_dir / "results.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {s.eval_dir / 'results.md'}")
    return 0


def cmd_doctor(args) -> int:
    """Check local prerequisites without downloading models or calling an API."""
    import importlib.util
    import os
    import platform

    s = get_settings()
    checks = []
    checks.append(("Python", platform.python_version()))
    modules = ["numpy", "faiss", "sentence_transformers", "transformers", "torch", "streamlit", "dotenv", "requests"]
    if s.llm_backend == "groq":
        modules.append("groq")
    for module in modules:
        checks.append((f"Python package: {module}", "installed" if importlib.util.find_spec(module) else "MISSING"))
    checks.append(("FAISS index", "present" if (s.index_dir / "faiss.index").exists() else "MISSING"))
    checks.append(("Chunk metadata", "present" if (s.index_dir / "chunks.jsonl").exists() else "MISSING"))
    checks.append(("LLM backend", s.llm_backend))
    if s.llm_backend == "groq":
        checks.append(("Groq API key", "configured" if os.getenv("GROQ_API_KEY") else "MISSING"))
    elif s.llm_backend == "ollama":
        checks.append(("Ollama model", s.ollama_model))
    failed = False
    print("Satyameter environment check\n")
    for name, status in checks:
        print(f"{'[OK]' if status not in ('MISSING',) else '[MISSING]'} {name}: {status}")
        failed = failed or status == "MISSING"
    print("\nNote: package presence does not guarantee model download or API connectivity.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="satyameter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download Wikipedia articles")
    p.add_argument("--limit", type=int, default=None, help="only the first N titles")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build-index", help="chunk, embed and save the index")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("search", help="retrieval only")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("check-llm", help="test the LLM connection")
    p.set_defaults(func=cmd_check_llm)

    p = sub.add_parser("ask", help="retrieve and generate a cited answer")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("verify", help="ask a question, then verify the answer claim by claim")
    p.add_argument("question")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("check-claim", help="verify a single statement")
    p.add_argument("claim")
    p.add_argument("-k", type=int, default=3)
    p.set_defaults(func=cmd_check_claim)

    p = sub.add_parser("eval-run", help="generate claims for hand-labelling")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--force", action="store_true", help="overwrite an existing labelling file")
    p.set_defaults(func=cmd_eval_run)

    p = sub.add_parser("eval-score", help="compute metrics from labelled claims")
    p.set_defaults(func=cmd_eval_score)

    p = sub.add_parser("doctor", help="check installed packages and project files")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
