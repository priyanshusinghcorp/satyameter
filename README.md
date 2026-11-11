# Satyameter

**An AI fact-verification and hallucination guardrail engine**

Satyameter improves a user query for retrieval, retrieves passages from a curated
Wikipedia-based knowledge base, generates an evidence-grounded answer, decomposes it
into factual claims, and checks each claim against retrieved passages using Natural
Language Inference (NLI). It presents the answer, sources, claim-level status and an
aggregate evidence-grounded score in a Streamlit dashboard.

> **Important:** Supported means the retrieved passage entails the claim. Contradicted
> means the NLI model detects conflicting evidence. Not Found means adequate support was
> not located. Not Found is not proof of falsity, and the Faithfulness Score measures
> evidence support—not absolute truth.

## Current implementation

- Query rewriting using the configured LLM, preserving the question's entities, dates,
  numbers and constraints.
- Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) and a FAISS inner-product index.
- Retrieval-Augmented Generation with numbered source passages.
- Pluggable LLM clients: Groq (recommended default for an 8 GB CPU-only laptop) and
  optional local Ollama.
- Atomic claim extraction and DeBERTa-v3 NLI classification.
- Supported / Contradicted / Not Found verdicts, with adjustable thresholds.
- SQLite response caching and local run/claim history.
- Streamlit dashboard, CLI tools and pytest suite.

## Architecture

```text
User question
   |
   v
Prompt rewriter --------------------> rewritten retrieval query
   |                                         |
   |                                         v
   |                                  Sentence-Transformer
   |                                         |
   |                                         v
   |                                     FAISS index
   |                                         |
   +-----------------------------------------+
                                             |
                                      top-k evidence
                                             |
                                             v
                                   LLM answer generation
                                             |
                                             v
                                      Claim extraction
                                             |
                                             v
                                    Per-claim retrieval
                                             |
                                             v
                                      NLI verification
                                             |
                                             v
                         Dashboard + score + citations + SQLite history
```

## Hardware guidance

The project is configured for CPU inference with a low thread count. The embedding and
NLI models run locally; the recommended default LLM backend is Groq because running a
larger generative model together with the NLI model can be slow on an entry-level dual-core
CPU with 8 GB RAM. Ollama is available as an optional local backend, but local generation
will be slower and should use a small model such as `llama3.2:1b` on this machine.

## Windows 11 setup

Use 64-bit **Python 3.12**. Open PowerShell in the extracted project folder.

1. Run the setup script:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\setup_windows.ps1
   ```

2. Open `.env` and add your own Groq API key:

   ```dotenv
   LLM_BACKEND=groq
   GROQ_API_KEY=your_actual_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   GROQ_REASONING_EFFORT=low
   ```

   Never upload `.env`, share the API key, or commit it to Git. The ZIP intentionally
   excludes `.env` and the Git metadata. Copy your previous `.env` into this project folder
   if you already configured Groq. To retain your existing Git history, copy the ZIP contents
   into your existing repository while preserving its `.git` directory and `.env`; otherwise
   initialise a new repository with `git init`.

3. Check the setup:

   ```powershell
   .\.venv\Scripts\python.exe -m satyameter.cli doctor
   ```

4. Start the dashboard:

   ```powershell
   .\run_app.bat
   ```

   Or run directly:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
   ```

The prebuilt corpus and FAISS index are included, so downloading Wikipedia or rebuilding
the index is not required for the first run. The first model use downloads the embedding
and NLI model weights from Hugging Face and therefore needs Internet access and disk space.

## Optional local LLM with Ollama

Install Ollama, then in a terminal run:

```powershell
ollama pull llama3.2:1b
```

Edit `.env`:

```dotenv
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT_S=180
```

Start the Ollama service before launching Satyameter. The local backend does not require a
Groq key. Performance depends on CPU, memory and model quantisation.

## CLI commands

Run commands from the project root with the virtual environment active:

```powershell
python -m satyameter.cli doctor
python -m satyameter.cli check-llm
python -m satyameter.cli search "Who discovered penicillin?"
python -m satyameter.cli ask "Who discovered penicillin and when?"
python -m satyameter.cli verify "Who discovered penicillin and when?"
python -m satyameter.cli check-claim "Alexander Fleming discovered penicillin in 1928."
python -m satyameter.cli eval-score
pytest -q
```

Optional corpus maintenance:

```powershell
python -m satyameter.cli fetch
python -m satyameter.cli build-index
```

Only rebuild the FAISS index if the corpus or embedding model changes. The configured
embedding model must match the model used to build the index; the application checks this.

## Evaluation data

`data/eval/claims_to_label.csv` currently contains 316 generated claim records with blank
`human_ok` values. These are not human-labelled ground truth. Review and label the claims
independently before running `eval-score`. The full procedure and metric interpretation are
in [`data/eval/README.md`](data/eval/README.md).

The score report distinguishes unsupported evidence (`NOT_FOUND` or `CONTRADICTED`)
from contradiction-only results. It avoids treating a missing passage as proof that a
statement is false. The old 316-row file predates the query rewriter; use `eval-run` to
collect a fresh comparison for the current pipeline. `eval-run` protects an existing CSV
from accidental overwriting unless `--force` is used.

## Project structure

```text
satyameter/
├── app/                      # Streamlit dashboard
├── data/
│   ├── raw/                  # Curated source articles
│   ├── index/                # Prebuilt FAISS index and chunk metadata
│   └── eval/                 # Questions and claim evaluation CSV
├── satyameter/
│   ├── evaluation/           # Evaluation generation and scoring
│   ├── generation/           # RAG answer generation
│   ├── ingestion/            # Wikipedia ingestion and chunking
│   ├── llm/                  # Groq, Ollama and response cache
│   ├── retrieval/             # Sentence-Transformer and FAISS
│   ├── verification/         # Claim extraction and NLI
│   ├── rewriter.py            # Conservative query rewriting
│   ├── storage.py             # SQLite run and claim history
│   ├── config.py              # Central configuration
│   └── pipeline.py            # End-to-end orchestration
├── tests/                     # Automated unit tests
├── setup_windows.ps1
├── run_app.bat
└── requirements.txt
```

## Testing

```powershell
pytest -q
```

The test suite uses fake LLM, embedder and NLI objects for most unit tests, so it does not
need API calls or model downloads. The FAISS integration test runs when `faiss-cpu` is
installed.

## Data attribution

The included source article text was retrieved from Wikipedia. Source titles and URLs are
retained in the article JSON files and chunk metadata. See [`data/CORPUS_ATTRIBUTION.md`](data/CORPUS_ATTRIBUTION.md)
for attribution and license notes.
