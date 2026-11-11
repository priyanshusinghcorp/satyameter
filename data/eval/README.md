# Evaluation data status

`claims_to_label.csv` contains 316 claim records from an earlier comparison of the
unguarded and RAG systems. Its `human_ok` field is blank for every row in this copy.
These records are **not human-labelled ground truth**, so `eval-score` correctly
refuses to report final performance until labels are entered.

Label convention:

- `1` = the claim is factually correct.
- `0` = the claim is factually incorrect or fabricated.
- blank = not reviewed / genuinely undecidable.

Review each claim against an appropriate independent source. Do not use the verifier's
own status as the ground-truth label. Preserve a backup before editing the CSV.

Run `python -m satyameter.cli eval-score` after saving the labelled CSV. The report
separately reports (a) claims not supported by retrieved evidence and (b) claims marked
Contradicted. `NOT_FOUND` means the configured corpus did not provide sufficient support;
it does not, by itself, prove a claim is false.

The 316 rows were generated before the prompt-rewriting stage was added and contain the
older `unguarded` and `rag` conditions. Re-run the benchmark with
`python -m satyameter.cli eval-run` to collect a fresh three-way ablation: `unguarded`,
`raw_query_rag`, and `rewritten_rag`. The new file also records the rewritten retrieval query.
The command will not overwrite an existing CSV unless `--force` is supplied.
