# Self-Hosting and Training a Reranker: A Community-Scale Pitfall Where a "Bad File" Masquerades as a "Weak Model"

> The three pillars of retrieval: embedding (recall) → **reranker (precision ranking)** → generation. This book covers self-hosted deployment of Qwen3-Reranker, our training-set pipeline, and a pitfall everyone should know about: **at the time (2026-07), multiple community-converted reranker GGUFs we downloaded were all missing the scoring head — scores were near-random, and no error was ever raised.**

## Update (2026-10)

The batch-size trap in the previously published command: setting only `--ubatch-size 16384` does not raise the logical batch limit. **Evidence level: source reading plus production logs, not executed.** The exact public command was not run, and the old/new live comparison described below remains open.

### Documented defaults and source reading

The installed build reported version `1 (6f3a9f3de)`. Its observed `llama-server --help` lists logical batch `-b` default **2048** and physical batch `-ub` default **512**. Source at exactly commit `6f3a9f3de`, read on **2026-10-03** without executing the public command, shows:

- `--reranking` enables embedding mode. In that mode, when the logical batch exceeds the physical batch, the server lowers the logical batch to the physical batch.
- Context creation sets the physical batch to the smaller of the logical and physical sizes. Thus `-ub 16384` with default `-b 2048` yields an effective physical batch of **2048**; the larger physical setting is clamped.
- Each query-plus-document pair must fit in one physical batch. An overlong pair is rejected with **HTTP 500**, not truncated: `input (N tokens) is too large to process. increase the physical batch size (current batch size: M)`. Production messages consistently reported the configured size: **2048** when configured at 2048 and **8192** when configured at 8192.
- The upstream help text documents `--cache-ram` default **8192 MiB**, with **0** disabling the prompt cache.

The previously published command therefore most likely behaves as a **2048-token** server. This conclusion rests on source reading, documented defaults, and consistent production behaviour; newer or older builds may differ.

### Production observations

These counts come from **one client log**, covering **2026-07-02 through 2026-09-22 (UTC)**. Each line represents one failed rerank request carrying **4 to 30 candidate documents**, not one failed document. Client source reading shows that failures fall back to the original, unreranked order, silently degrading ranking.

| Period (UTC) | Effective batch in message | Logged failures | Rejected input size (tokens) |
|---|---|---|---|
| 2026-07-02 | 512 | 2 | 1,658 to 1,770 |
| 2026-07-03 to 2026-07-05 | 8192 | 248 | 9,636 to 54,851 (median 33,669) |
| 2026-07-06 to 2026-09-22 | 2048 | 49,482 over 66 distinct days (peak 956 on one day) | 2,050 to 54,858 (median 2,492, 90th percentile 3,019) |

All but **7** of the **49,482** failures in the 2048 period involved inputs at or below **4,096 tokens**. A 4096 limit would have removed this batch-limit cause for **more than 99.98 percent** of those failures, computed from the log; this is not a measured post-change success rate. The 2048 failures spanned **eleven weeks**. Later entries in the same log place the last 2048-message failure on **2026-09-24**. Recorded configuration changed both batch sizes to **4096 on 2026-09-25**; that configuration was not re-read live. Later failures use a different message, discussed below.

### Configuration history and memory conditions

The production server used a **self-fine-tuned 0.6B reranker in Q8_0** with llama.cpp. Up to **2026-07-05**, the configuration file specified `-c 32768 -b 32768 -ub 32768`, but the running process used `-b 8192 -ub 8192` with no `-c` (model default context). The file and process had drifted apart.

On **2026-07-05/06**, the configuration changed to `-c 4096 -b 2048 -ub 2048`, based on **13,499 chunks** with mean **388 tokens**, 95th percentile **948**, maximum **1,500**, and at most **50 candidates per request**. This assumed each rerank input was one chunk. Rejected inputs of **2,050 to 54,858 tokens** show that assumption did not hold for all callers; the callers responsible were not diagnosed.

Physical footprint measured once with macOS `vmmap` on an **Apple M4 Max Mac with 64 GB** fell from **19.4 GB (peak 22.2 GB)** with `-b/-ub 8192` and default context to **499.3 MB** with `-c 4096 -b/-ub 2048`. Batch and context changed together, so their individual effects cannot be separated. This was one observation, not a controlled comparison.

Derive batch and context sizes from the real maximum query-plus-document pair across every caller, cap document length in the client, and inspect the server log for the rejection message. Client-side truncation is a recommendation, not a verified fix in these observations.

### Deploy correction and open verification

The old `scripts/deploy.sh` supplied only `--ubatch-size 16384`. Its two short documents checked relevant above irrelevant and not near zero, so the assertion could pass despite the long-input failure. The corrected script shares one `--max-tokens` value across logical batch, physical batch, and context, defaulting to **4096**, the recorded production batch value. It prints `n_batch` and `n_ubatch` from the startup log and adds a second request with one document of approximately **3,000 tokens**, requiring **HTTP 200** and a finite score. The exact token count depends on the tokenizer.

Open / not verified:

- The public command itself and the live old/new comparison were **not run**. On build `6f3a9f3de` or newer, the intended check is that the old command fails the long-document request with HTTP 500 and `too large to process`, while the corrected command passes. Offline argument and mocked-response tests do not establish either live result.
- After the 4096 change, the same log contains a handful of `Context size has been exceeded` failures. Source reading associates this message with decoding finding no room in the context, distinct from the overlong-input rejection. The cause remains undiagnosed. No rejection above **4,096 tokens** was recorded in the examined log; such rejection is expected from the configuration but was not observed.
- Only one client's log was examined; a second client on a different system was not examined.
- Larger shared values for the corrected command were not memory-tested. In particular, **16384** for batch and context was never memory-measured and is not a recommendation. Memory cost grows with batch and context sizes.
- Production also applied `--cache-ram 0`, followed by a sharp resident-memory drop. That change coincided with a restart, was measured once, and was not isolated; no numerical result or causal attribution is established here.
- Which callers send tens of thousands of tokens remains undiagnosed; client-side truncation remains unverified.

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

Set `RERANK_BASE_URL` to the intended server's base URL, then run `scripts/deploy.sh --size 0.6B --base-url "$RERANK_BASE_URL"` (or use `--size 4B`). It downloads official weights → converts with a recent converter → **automatic cls.output.weight check** → starts the server → prints effective batch sizes → checks short-document scoring and long-document acceptance. Models default to `./models`; `--models-dir DIR` overrides it. The supplied base URL must reach the same server and match `--port` if overridden.

`--max-tokens N` sets all three limits together; the default is **4096**. `scripts/deploy.sh --dry-run` prints the generated command without checking dependencies, downloading a model, or starting a server. An already-running instance is reused, so changing this option requires restarting that instance; inspect the printed log values to confirm its settings.

## Deployment (llama.cpp server)

```bash
MAX_TOKENS=4096
llama-server -m '<self-converted-with-head.gguf>' --reranking --pooling rank \
  -c "$MAX_TOKENS" -b "$MAX_TOKENS" -ub "$MAX_TOKENS"
```
- `-b` (logical batch, default **2048**) and `-ub` (physical batch, default **512**) must both be raised. The server clamps physical batch to logical batch, so `-ub 16384` alone is effectively **2048**, and any query-plus-document pair longer than the effective physical batch is rejected with **HTTP 500**, not truncated (source reading at `6f3a9f3de`, not a live test). Set batch and context at least as large as the longest pair, cap document length in the client, and measure memory: both sizes cost memory. **4096** is the recorded production batch value; larger shared values for this corrected command were not measured and are not recommended without measurement.
- Wiring note: **fire a real request and assert the score distribution looks healthy** (relevant above irrelevant and not near zero). A short pair cannot catch the batch-size problem: also send one document of approximately **3,000 tokens** and require **HTTP 200** with a finite score. The deploy script includes both assertions; live results remain unverified.

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
*Updated 2026-10. Evidence and measurement conditions are stated above. Issues welcome.*
