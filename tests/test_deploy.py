import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts.check_rerank import check_scoring


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts" / "deploy.sh"
BASH = shutil.which("bash")


class DeployTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix=".deploy-test-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_deploy(self, *args, env=None):
        return subprocess.run(
            [BASH, str(DEPLOY), *args], cwd=self.directory,
            env=env, text=True, capture_output=True, timeout=15,
        )

    def dry_run(self, *args):
        # An empty PATH proves that dry-run needs no server, model tools, or Python.
        result = self.run_deploy("--dry-run", *args, env={**os.environ, "PATH": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])
        return shlex.split(result.stdout)

    def test_default_command_shares_4096(self):
        self.assertEqual(self.dry_run(), [
            "llama-server", "-m", "./models/qwen3-reranker-0.6B/model-f16.gguf",
            "--reranking", "--pooling", "rank", "--batch-size", "4096",
            "--ubatch-size", "4096", "--ctx-size", "4096", "--port", "8081",
        ])

    def test_override_updates_all_limits_and_preserves_spaces(self):
        command = self.dry_run("--max-tokens", "3072", "--models-dir", "model files", "--size", "4B")
        for option in ("--batch-size", "--ubatch-size", "--ctx-size"):
            self.assertEqual(command.count(option), 1)
            self.assertEqual(command[command.index(option) + 1], "3072")
        self.assertEqual(command[command.index("-m") + 1], "model files/qwen3-reranker-4B/model-f16.gguf")

    def test_invalid_token_limits_fail_before_side_effects(self):
        for value in ("0", "-1", "1.5", "invalid", "", "000"):
            with self.subTest(value=value):
                result = self.run_deploy("--dry-run", "--max-tokens", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be a positive integer", result.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_option_value_fails(self):
        result = self.run_deploy("--dry-run", "--max-tokens")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing value for --max-tokens", result.stderr)

    def test_live_mode_requires_explicit_base_url(self):
        result = self.run_deploy(env={**os.environ, "PATH": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--base-url is required", result.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])

    def stub(self, name, body):
        executable = self.directory / "bin" / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)

    def test_startup_prints_effective_sizes_and_invokes_assertions(self):
        # Stub all server, model, and HTTP operations; no live deployment occurs.
        (self.directory / "bin").mkdir()
        work = self.directory / "models" / "qwen3-reranker-0.6B"
        work.mkdir(parents=True)
        (work / "model-f16.gguf").write_text("offline fixture")
        self.stub("llama-server", '''
printf '%s\\n' "$@" > "$TEST_WORK/server.args"
printf 'llama_context: n_batch = 2048\\nllama_context: n_ubatch = 512\\n'
touch "$TEST_WORK/ready"
''')
        self.stub("python3", '''
case "$1" in
  -c|-) exit 0;;
  *) printf '%s\\n' "$@" > "$TEST_WORK/assertion.args";;
esac
''')
        self.stub("curl", '''
for i in $(seq 1 100); do
  [ -f "$TEST_WORK/ready" ] && exit 0
  sleep 0.01
done
exit 1
''')
        self.stub("lsof", "exit 1")
        self.stub("hf", 'touch "$TEST_WORK/unexpected-download"; exit 1')
        result = self.run_deploy("--base-url", "offline", env={
            **os.environ, "TEST_WORK": str(work),
            "PATH": str(self.directory / "bin") + os.pathsep + os.environ.get("PATH", ""),
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("n_batch = 2048", result.stdout)
        self.assertIn("n_ubatch = 512", result.stdout)
        self.assertEqual((work / "assertion.args").read_text().splitlines(), [
            str(ROOT / "scripts" / "check_rerank.py"), "offline",
        ])
        arguments = (work / "server.args").read_text().splitlines()
        for option in ("--batch-size", "--ubatch-size", "--ctx-size"):
            self.assertEqual(arguments[arguments.index(option) + 1], "4096")
        self.assertFalse((work / "unexpected-download").exists())


class ScoringTests(unittest.TestCase):
    # No authority is needed: urlopen is always mocked before this URL is used.
    base_url = "http:///offline"

    def response(self, scores, status=200, indices=None):
        if indices is None:
            indices = range(len(scores))
        response = io.BytesIO(json.dumps({"results": [
            {"index": index, "relevance_score": score}
            for index, score in zip(indices, scores)
        ]}).encode())
        response.status = status
        return response

    def check(self, responses):
        with patch("scripts.check_rerank.urllib.request.urlopen", side_effect=responses) as opener:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                check_scoring(self.base_url)
        return opener, output.getvalue()

    def test_short_and_long_requests_pass(self):
        opener, output = self.check([
            self.response([0.1, 0.9], indices=[1, 0]), self.response([0.5]),
        ])
        self.assertEqual(opener.call_count, 2)
        short, long = [json.loads(call.args[0].data) for call in opener.call_args_list]
        self.assertEqual(len(short["documents"]), 2)
        self.assertEqual(len(long["documents"]), 1)
        self.assertGreater(len(long["documents"][0].split()), 2048)
        self.assertIn("long document: HTTP 200, finite score=0.5000", output)

    def test_long_http_500_is_not_hidden_by_short_success(self):
        error = urllib.error.HTTPError(self.base_url, 500, "too large to process", {}, None)
        with self.assertRaisesRegex(urllib.error.HTTPError, "too large to process"):
            self.check([self.response([0.9, 0.1]), error])

    def test_long_response_must_be_http_200(self):
        with self.assertRaisesRegex(AssertionError, "expected HTTP 200, got 202"):
            self.check([self.response([0.9, 0.1]), self.response([0.5], status=202)])

    def test_long_score_must_be_finite(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaisesRegex(AssertionError, "non-finite score"):
                self.check([self.response([0.9, 0.1]), self.response([score])])

    def test_long_response_must_contain_one_score(self):
        with self.assertRaisesRegex(AssertionError, "missing or duplicate scores"):
            self.check([self.response([0.9, 0.1]), self.response([])])

    def test_short_scoring_assertions_remain_enforced(self):
        for scores, message in (([0.1, 0.9], "does not outscore"), ([1e-23, 0.0], "near-zero")):
            with self.subTest(scores=scores), self.assertRaisesRegex(AssertionError, message):
                self.check([self.response(scores)])


if __name__ == "__main__":
    unittest.main()
