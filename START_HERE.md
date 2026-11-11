# Satyameter — first run and teacher demonstration

## Before starting

- Keep your existing `.env` file (it contains your private configuration/API key) somewhere safe. The project ZIP does not include it.
- Extract this project into a normal folder such as `C:\dev\satyameter`.
- Use 64-bit Python 3.12.
- Keep an Internet connection available for package/model downloads and Groq requests.

## First-time setup (Windows 11)

Open PowerShell in the extracted `satyameter` folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

If the setup script creates a new `.env`, open it and set your Groq API key. If you already have a valid `.env` from your previous project folder, copy that file into this folder instead. Do not upload or share it.

Check the environment:

```powershell
.\.venv\Scripts\python.exe -m satyameter.cli doctor
.\.venv\Scripts\python.exe -m satyameter.cli check-llm
```

Start the dashboard:

```powershell
.\run_app.bat
```

The first dashboard run downloads the embedding and NLI model weights. The prebuilt FAISS index and curated article corpus are already included, so do not run `fetch` or `build-index` for the first demonstration.

## Suggested 3-minute demonstration

1. **In-scope question:** `Who discovered penicillin and when?` Show the rewritten query, generated answer, source passages and claim-by-claim labels.
2. **Another in-scope question:** `Who created the Python programming language?` Open the retrieved source and show how the claim maps to evidence.
3. **Out-of-scope / unsupported question:** `What is the capital of Atlantis?` Explain that a `NOT_FOUND` result means the configured knowledge base did not provide support; it is not itself proof that the claim is false.
4. Change the support threshold in the sidebar to demonstrate that the current NLI scores can be reclassified without calling the models again.
5. Open **Recent local run history** to show that run summaries and claim evidence are stored in SQLite.

## What to explain to the teacher

- Satyameter is a middleware prototype: it rewrites the retrieval query, retrieves source passages, generates an answer, extracts atomic claims and checks each claim using Natural Language Inference.
- Supported, Contradicted and Not Found are evidence-relationship labels.
- Faithfulness Score = supported claims divided by total extracted claims. It measures support from retrieved evidence, not absolute truth.
- The system uses a curated 59-article corpus (4,967 indexed chunks in the supplied index). Its coverage is intentionally bounded.
- Groq is the recommended LLM backend for this CPU-only laptop; Ollama is an optional local backend. The embedding and NLI stages run locally.
- The supplied 316-claim evaluation CSV has blank human labels. Do not present final precision, recall, F1 or hallucination-rate results until the claims have been independently reviewed.

## If something fails

- `GROQ_API_KEY is not set`: confirm `.env` is in the project root and the key is on the `GROQ_API_KEY=` line.
- `No index found` or incomplete index: check that `data/index/faiss.index`, `data/index/chunks.jsonl` and `data/index/meta.json` exist.
- Slow first run: model downloads and first model loading take longer than later runs. Do not rebuild the index unless the embedding model or corpus changes.
- Model/API connection error: confirm Internet access and run `python -m satyameter.cli check-llm` from the activated environment.
- Package import error: run the setup script again from the project root.
