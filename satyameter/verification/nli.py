"""Natural Language Inference: does the evidence entail, contradict, or say nothing about a claim?"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


def map_labels(id2label: dict) -> dict[str, int]:
    """Map a model's label names to class indexes without assuming their order."""
    mapping: dict[str, int] = {}
    for index, name in id2label.items():
        key = str(name).lower()
        if "entail" in key:
            mapping["entailment"] = int(index)
        elif "contra" in key:
            mapping["contradiction"] = int(index)
        elif "neutral" in key:
            mapping["neutral"] = int(index)
    missing = {"entailment", "contradiction", "neutral"} - set(mapping)
    if missing:
        raise ValueError(f"Model labels {id2label} are missing: {sorted(missing)}")
    return mapping


class NLIModel:
    """Lazy-loaded Hugging Face NLI cross-encoder (premise = evidence, hypothesis = claim)."""

    def __init__(
        self, model_name: str, batch_size: int = 4, max_length: int = 384, cpu_threads: int = 2
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cpu_threads = max(1, int(cpu_threads))
        self._tokenizer = None
        self._model = None
        self._labels: dict[str, int] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Avoid excessive thread creation on entry-level dual-core CPUs.
        torch.set_num_threads(self.cpu_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        self._labels = map_labels(self._model.config.id2label)

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """pairs are (evidence, claim). Returns one NLIScores per pair, same order."""
        import torch

        if not pairs:
            return []
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16 tokens.")
        self._load()
        results: list[NLIScores] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = self._tokenizer(
                [evidence for evidence, _ in batch],
                [claim for _, claim in batch],
                return_tensors="pt",
                padding=True,
                truncation="only_first",  # cut the evidence, never the claim
                max_length=self.max_length,
            )
            with torch.no_grad():
                probs = torch.softmax(self._model(**inputs).logits, dim=-1)
            for row in probs:
                results.append(
                    NLIScores(
                        entailment=float(row[self._labels["entailment"]]),
                        contradiction=float(row[self._labels["contradiction"]]),
                        neutral=float(row[self._labels["neutral"]]),
                    )
                )
        return results
