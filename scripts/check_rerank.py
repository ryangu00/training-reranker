"""Check short-document scoring and long-document acceptance after deployment."""

import json
import math
import sys
import urllib.request


def request_scores(base_url, documents):
    payload = {"model": "m", "query": "What is speculative decoding", "documents": documents}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/rerank",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200, f"FATAL: expected HTTP 200, got {response.status}"
        results = sorted(json.load(response)["results"], key=lambda item: item["index"])
    assert [item["index"] for item in results] == list(range(len(documents))), "FATAL: missing or duplicate scores"
    scores = [item["relevance_score"] for item in results]
    assert all(math.isfinite(score) for score in scores), "FATAL: non-finite score"
    return scores


def check_scoring(base_url):
    scores = request_scores(base_url, [
        "Speculative decoding drafts tokens with a small model and verifies them with the large model, speeding up generation.",
        "Today's lunch is pasta.",
    ])
    print(f"  relevant={scores[0]:.4f} irrelevant={scores[1]:.4f}")
    assert scores[0] > scores[1], "FATAL: relevant document does not outscore the irrelevant one - scoring chain is broken"
    assert scores[0] > 1e-10, "FATAL: near-zero score (the 4.5e-23 disease), check the scoring head"

    # About 3,000 tokens; the exact length depends on the model tokenizer.
    document = ("The small model drafts tokens for the larger model. " * 300).strip()
    scores = request_scores(base_url, [document])
    print(f"  long document: HTTP 200, finite score={scores[0]:.4f}")


if __name__ == "__main__":
    check_scoring(sys.argv[1])
