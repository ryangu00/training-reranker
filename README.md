# Self-Hosting and Training a Reranker: A Community-Scale Pitfall Where a "Bad File" Masquerades as a "Weak Model"

> The three pillars of retrieval: embedding (recall) → **reranker (precision ranking)** → generation. This book covers self-hosted deployment of Qwen3-Reranker, our training-set pipeline, and a pitfall everyone should know about: **at the time (2026-07), multiple community-converted reranker GGUFs we downloaded were all missing the scoring head — scores were near-random, and no error was ever raised.**

## The Big Pitfall First: GGUF Missing `cls.output.weight`

**Symptom**: llama.cpp in `--reranking` mode runs perfectly fine, but every document scores near-zero values like 4.5e-23. Wired into the retrieval chain, reranking makes results **worse than no reranker at all** (measured on our in-domain holdout set: recall@1 dropped from 0.5667 to 0.0417 — below random: picking randomly out of k candidates has expectation ≈1/k, so it was confidently ranking the right answers down).
**Root cause**: these GGUFs are missing `cls.output.weight` (the scoring-head tensor) — the conversion tools in use during that period did not recognize this head and silently dropped it (later versions have fixed this, hence you must convert yourself with a recent version). Without the head, `--reranking` has no scoring projection, outputs garbage values, and **raises no error**.
**Diagnosis** (30 seconds):
```bash
python3 -c "
from gguf import GGUFReader
r = GGUFReader('<model.gguf>')
print([t.name for t in r.tensors if 'cls' in t.name])"
# Empty list = bad file; you should see cls.output.weight etc.
```
**Fix**: skip community GGUFs. Convert the official HF weights (e.g. `Qwen/Qwen3-Reranker-0.6B`/`-4B`) yourself with a **recent** llama.cpp `convert_hf_to_gguf.py` — the artifact with the head has a visibly different tensor count (with the version we used, 0.6B: 311 tensors / 4B: 399, usable as a verification fingerprint; other versions may differ, the core check is that the cls tensors must be present).

## One-Command Deploy

`scripts/deploy.sh --size 0.6B|4B` — downloads official weights → converts with a recent converter → **automatic cls.output.weight check** (this book's big pitfall, mechanized) → starts the server → real scoring assertion (relevant > irrelevant, and not near-zero).

## Deployment (llama.cpp server)

```bash
llama-server -m <self-converted-with-head.gguf> --reranking --pooling rank --ubatch-size 16384
```
- `--ubatch-size 16384`: preventive setting — ensures long document pairs (thousands of tokens) are not truncated at batch boundaries (same lesson as num_batch in the embedding book).
- Wiring note: the upstream reads the rerank endpoint field from its config — **fire one real request and assert the score distribution looks healthy** (a good model's scores show clear separation, not all crowded near 0 or 1). Don't just verify the service returns 200.

## Results Methodology (Our Three-Way Comparison)

| Setup | Retrieval quality gain (our holdout set) |
|---|---|
| No reranker (pure embedding recall) | baseline |
| +Qwen3-Reranker-0.6B (self-converted, with head) | about +4 points |
| +Qwen3-Reranker-4B (self-converted, with head) | about +11 points |

The 4B gain is markedly higher; 0.6B wins on latency. Pick by your latency budget. (Measurement setup note: this is an internal composite retrieval score on our in-domain holdout set — absolute values don't compare across environments, but the **direction and relative magnitude** are a transferable reference.)

## Training (If You Need Further Specialization)

The reranker training-set schema comes from the same pipeline as the embedding book (query + positive/negative passage pairs); the data volume requirement is larger than for embeddings (MB-scale JSONL). Practical advice: **start with official weights plus a self-converted head fix, confirm the reranking chain is healthy, and only then talk about training** — our biggest win came from "swapping a bad file for a good one" (0.0417 → 0.5667+); training specialization is incremental gain on top of that. Training data is not shipped with the repo (contains private corpora); the schema and generation scripts are the same as the embedding book.

## Reusable Conclusions

- **File-level failures masquerading as a weak model are the most dangerous kind**: no crash, no error, just bad metrics. For any downloaded quantized artifact, first run a tensor inventory check (missing head/layers), then a behavioral assertion (score distribution / known examples), and only then judge "whether the model is good".
- The go-live assertion trio for a reranking chain: tensor fingerprint ✓ / real-request score distribution ✓ / holdout comparison against baseline ✓.

---
*RyanAI Lab · All numbers measured on our resident environment. Updated 2026-09. Issues welcome.*
