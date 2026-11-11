# Satyameter implementation changes

## Updated project package

- Added a conservative prompt-rewriting stage before retrieval. The original question is still sent to answer generation; the rewritten query is used for retrieval and displayed in the dashboard.
- Added citation validation so a generated `[n]` marker is removed when that passage number was not supplied to the model.
- Added an optional Ollama HTTP backend while retaining Groq as the default. This supports both a practical hosted backend for a low-resource laptop and a local alternative.
- Added SQLite persistence for run summaries and claim/evidence records.
- Reduced CPU thread use and NLI batch size for a modest Windows laptop.
- Added vector-index integrity checks for embedding model, dimensions and chunk counts.
- Improved evaluation reporting to distinguish unsupported evidence from contradiction-only classifications. `NOT_FOUND` is not treated as proof of factual falsity.
- Expanded the evaluation harness to compare unguarded generation, RAG with the raw question and RAG with the rewritten question.
- Added Windows setup/start scripts, a CLI environment check, attribution notes, and a first-run/demo guide.
- Corrected `.gitignore` so the bundled source corpus and prebuilt index can be committed with the application.
- Excluded `.env`, `.git`, local SQLite caches and Python bytecode from the distributable ZIP.

## Validation performed in the packaging environment

- Python syntax compilation: passed.
- Automated tests: 51 passed.
- FAISS index integration test: skipped because FAISS could not be installed in the packaging environment (no package-network access).
- Evaluation scoring command: correctly refuses to report metrics while `human_ok` is blank for all 316 records.
- Live Groq/Ollama generation and model downloads were not executed in this packaging environment. They must be checked on the target Windows laptop after installing dependencies and configuring the backend.
