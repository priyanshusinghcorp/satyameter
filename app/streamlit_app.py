"""Satyameter web dashboard. Run with: python -m streamlit run app/streamlit_app.py"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from satyameter.config import get_settings
from satyameter.storage import SQLiteStore
from satyameter.verification.verifier import Status, Thresholds, is_negated, rethreshold

st.set_page_config(
    page_title="Satyameter | Evidence-first AI",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud secrets are supported, while local development uses .env.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

settings = get_settings()

STATUS_META = {
    Status.SUPPORTED: {"label": "SUPPORTED", "color": "#16865b", "icon": "✓"},
    Status.CONTRADICTED: {"label": "CONTRADICTED", "color": "#c44747", "icon": "×"},
    Status.NOT_FOUND: {"label": "NOT FOUND", "color": "#a87519", "icon": "?"},
}
EXAMPLES = [
    "Who discovered penicillin and when?",
    "Who created the Python programming language?",
    "What did Marie Curie win Nobel Prizes for?",
    "What is the highest mountain on Earth and how tall is it?",
    "What was the Apollo 11 mission?",
    "When did Albert Einstein win the Nobel Prize in Chemistry?",
    "What is the capital of Atlantis?",
]


@st.cache_resource(show_spinner="Loading the retrieval, language and verification models…")
def load_pipeline():
    from satyameter.pipeline import Satyameter

    return Satyameter()


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render_verdict(verdict) -> None:
    meta = STATUS_META[verdict.status]
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:10px 0 5px'>"
        f"<span style='display:inline-block;background:{meta['color']};color:white;"
        f"padding:4px 9px;border-radius:5px;font-size:11px;font-weight:700;letter-spacing:.4px'>"
        f"{meta['icon']} {meta['label']}</span>"
        f"<span style='font-weight:550'>{html.escape(verdict.claim)}</span></div>",
        unsafe_allow_html=True,
    )

    if verdict.status is Status.CONTRADICTED and is_negated(verdict.claim):
        st.warning(
            "This claim contains a negation. NLI models can misread absence and negative wording; "
            "inspect the evidence before treating this as a definite falsehood."
        )

    if verdict.evidence:
        with st.expander(f"Inspect evidence ({len(verdict.evidence)} passages)"):
            for number, evidence in enumerate(verdict.evidence, start=1):
                chunk, scores = evidence.chunk, evidence.scores
                st.markdown(
                    f"**Passage {number}: {html.escape(chunk.doc_title)} — "
                    f"{html.escape(chunk.section)}**  ·  "
                    f"[Open source]({chunk.url})"
                )
                st.caption(
                    f"Entailment {scores.entailment:.3f}  ·  "
                    f"Contradiction {scores.contradiction:.3f}  ·  Neutral {scores.neutral:.3f}"
                )
                st.write(chunk.text)
                if number != len(verdict.evidence):
                    st.divider()
    elif verdict.best is None:
        st.caption("No evidence passage was retrieved for this claim.")


# Sidebar controls
with st.sidebar:
    st.markdown("## SATYAMETER")
    st.caption("Evidence-first AI answer inspection")
    st.divider()
    answer_mode = st.radio(
        "Generation mode",
        ["Guarded RAG", "Unguarded baseline"],
        help="The baseline answers without retrieved context. Both modes are checked against the knowledge base.",
    )
    rewrite_enabled = st.checkbox(
        "Enable prompt rewriting", value=True, disabled=answer_mode != "Guarded RAG",
        help="Rewrites the retrieval query while preserving names, dates and constraints.",
    )
    st.markdown("**Verification thresholds**")
    entail = st.slider("Support threshold", 0.50, 0.99, 0.70, 0.01)
    contradict = st.slider("Contradiction threshold", 0.70, 0.999, 0.90, 0.001, format="%.3f")
    ceiling = st.slider("Maximum entailment for contradiction", 0.05, 0.50, 0.20, 0.05)
    st.caption("Threshold changes re-score the current claims without another model call.")
    st.divider()
    st.markdown("**Runtime configuration**")
    st.caption(f"LLM backend: `{settings.llm_backend}`")
    st.caption(f"Embedding model: `{settings.embed_model}`")
    st.caption(f"NLI model: `{settings.nli_model}`")
    st.caption("Optimised for CPU inference")

# Page header
st.markdown(
    "<div style='padding:8px 0 2px'>"
    "<div style='font-size:12px;letter-spacing:2px;font-weight:700;color:#5f8fba'>"
    "TRUST &amp; VERIFICATION LAYER</div>"
    "<h1 style='margin:4px 0 2px;font-size:38px'>Satyameter</h1>"
    "<div style='color:#8b929d;font-size:16px'>Inspect what an AI answer says, what the evidence supports, and what remains uncertain.</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

left, right = st.columns([3, 1])
with left:
    st.markdown("### Ask a factual question")
    example = st.selectbox("Try a sample question", ["Choose an example…"] + EXAMPLES, label_visibility="collapsed")
    question = st.text_area(
        "Your question",
        value="" if example == "Choose an example…" else example,
        placeholder="e.g. Who discovered penicillin and when?",
        height=92,
        label_visibility="collapsed",
    )
with right:
    st.markdown("### Processing pipeline")
    st.markdown(
        "<div style='font-size:13px;line-height:1.9;color:#8994a3'>"
        "01&nbsp; Query rewrite<br>02&nbsp; Evidence retrieval<br>03&nbsp; Answer generation<br>"
        "04&nbsp; Claim extraction<br>05&nbsp; NLI verification"
        "</div>",
        unsafe_allow_html=True,
    )

run_clicked = st.button("Run verification", type="primary", disabled=not question.strip(), use_container_width=True)
if run_clicked:
    try:
        with st.spinner("Processing the query and checking factual claims…"):
            started = time.perf_counter()
            pipe = load_pipeline()
            answer, verification = pipe.ask(
                question.strip(),
                guarded=answer_mode == "Guarded RAG",
                rewrite=rewrite_enabled,
            )
            total_elapsed = time.perf_counter() - started
            run_id = SQLiteStore(settings.history_path).save_run(
                answer, verification, answer_mode
            )
        st.session_state["answer"] = answer
        st.session_state["verification"] = verification
        st.session_state["mode"] = answer_mode
        st.session_state["total_elapsed"] = total_elapsed
        st.session_state["run_id"] = run_id
    except FileNotFoundError as exc:
        st.error(f"Project file is missing: {exc}")
        st.info("Check the bundled data/index folder. Rebuild the index with `python -m satyameter.cli build-index` if necessary.")
    except Exception as exc:
        st.error(f"The request could not be completed: {exc}")
        st.info(
            "Check that the virtual environment is active, dependencies are installed, "
            "the configured LLM service is running, and the API key is present in .env when using Groq."
        )

if "verification" in st.session_state:
    answer = st.session_state["answer"]
    original = st.session_state["verification"]
    result = rethreshold(
        original,
        Thresholds(entail=entail, contradict=contradict, entail_ceiling=ceiling),
    )

    st.divider()
    st.markdown("### Verification overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evidence-grounded score", percent(result.faithfulness))
    m2.metric("Contradicted claims", percent(result.contradiction_rate))
    m3.metric("Decidable claims", percent(result.coverage))
    m4.metric("Total claims", len(result.verdicts))

    if result.faithfulness is not None:
        st.progress(result.faithfulness, text="Share of answer claims supported by retrieved evidence")

    st.markdown("### Generated answer")
    if answer.rewritten_query:
        with st.expander("Inspect prompt rewriting"):
            st.caption("Original question")
            st.write(answer.question)
            st.caption("Rewritten retrieval query")
            st.info(answer.rewritten_query)
    st.info(answer.answer or "(The language model returned an empty answer.)")

    st.markdown("### Claim-level assessment")
    if not result.verdicts:
        st.info("No assessable factual claims were extracted. The answer may be empty or the knowledge base may not contain the requested information.")
    else:
        for verdict in result.verdicts:
            with st.container(border=True):
                render_verdict(verdict)

    st.markdown("### Retrieved sources")
    if answer.hits:
        for i, hit in enumerate(answer.hits, start=1):
            chunk = hit.chunk
            with st.expander(f"[{i}] {chunk.doc_title} — {chunk.section}  |  similarity {hit.score:.3f}"):
                st.write(chunk.text)
                st.markdown(f"[Open source article]({chunk.url})")
    else:
        st.caption("No passages were retrieved.")

    st.caption(
        f"Mode: {st.session_state.get('mode', '—')}  ·  Model: {answer.model}  ·  "
        f"Total runtime: {st.session_state.get('total_elapsed', 0):.1f}s  ·  "
        f"Verification run ID: {st.session_state.get('run_id', '—')}"
    )
    st.warning(
        "Interpretation: Supported means the retrieved passage entails the claim. "
        "Contradicted means the model detected conflicting evidence. Not Found means adequate "
        "support was not located. Not Found does not prove a claim is false, and this score is "
        "evidence-grounded faithfulness—not a guarantee of truth."
    )

# Local run history
with st.expander("Recent local run history"):
    try:
        history = SQLiteStore(settings.history_path).recent_runs(10)
        if history:
            st.dataframe(
                [
                    {
                        "Run": row["run_id"],
                        "Date (UTC)": row["created_at"],
                        "Mode": row["mode"],
                        "Question": row["question"],
                        "Grounded score": percent(row["faithfulness"]),
                        "Claims": len(SQLiteStore(settings.history_path).claims_for_run(row["run_id"])),
                    }
                    for row in history
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Completed runs will appear here. History is stored locally in data/history.sqlite.")
    except Exception as exc:
        st.caption(f"History is not available: {exc}")

st.divider()
st.caption(
    "Satyameter is an academic research prototype. It provides evidence-based decision support; "
    "users should inspect the cited passages and independently validate important claims."
)
