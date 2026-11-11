"""One object that wires the whole system together. The CLI, dashboard and evaluation all use it."""

from __future__ import annotations

from satyameter.config import Settings, get_settings
from satyameter.generation.rag import RagAnswer, answer_question
from satyameter.llm import get_llm
from satyameter.retrieval.index import SentenceTransformerEmbedder, VectorIndex
from satyameter.verification.nli import NLIModel
from satyameter.verification.verifier import Thresholds, Verification, verify_answer

UNGUARDED_SYSTEM = (
    "You are a helpful assistant. Answer the question concisely "
    "(at most five sentences) from your own knowledge."
)


class Satyameter:
    def __init__(self, settings: Settings | None = None, index=None, llm=None, nli=None):
        self.settings = settings or get_settings()
        s = self.settings
        self.index = index or VectorIndex.load(s.index_dir, SentenceTransformerEmbedder(s.embed_model))
        self.llm = llm or get_llm(s)
        self.nli = nli or NLIModel(
            s.nli_model, max_length=s.nli_max_length, cpu_threads=s.cpu_threads
        )

    def answer_unguarded(self, question: str) -> RagAnswer:
        """Baseline: the LLM answers from memory, with no retrieval."""
        reply = self.llm.generate(UNGUARDED_SYSTEM, question, temperature=0.0, max_tokens=1024)
        return RagAnswer(question, reply, [], self.llm.name)

    def answer_rag(self, question: str, rewrite: bool = True) -> RagAnswer:
        return answer_question(
            question, self.index, self.llm, k=self.settings.top_k, rewrite=rewrite
        )

    def verify(self, answer: RagAnswer, thresholds: Thresholds = Thresholds()) -> Verification:
        s = self.settings
        return verify_answer(
            answer, self.index, self.llm, self.nli, thresholds,
            claim_k=s.claim_k, max_evidence=s.max_evidence,
        )

    def ask(
        self,
        question: str,
        guarded: bool = True,
        thresholds: Thresholds = Thresholds(),
        rewrite: bool = True,
    ):
        """Return (answer, verification); optionally disable rewriting for ablation tests."""
        answer = self.answer_rag(question, rewrite=rewrite) if guarded else self.answer_unguarded(question)
        return answer, self.verify(answer, thresholds)
