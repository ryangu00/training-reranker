#!/usr/bin/env bash
# deploy.sh — Qwen3-Reranker 自托管一键部署(含"缺打分头"自动检查=本书大坑的机器化)
# 用法: ./deploy.sh [--size 0.6B|4B] [--port 8081] [--models-dir ~/models]
set -euo pipefail
SIZE="0.6B"; PORT=8081; MODELS_DIR="$HOME/models"
while [ $# -gt 0 ]; do case "$1" in
  --size) SIZE="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  --models-dir) MODELS_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done
HF_REPO="Qwen/Qwen3-Reranker-$SIZE"
WORK="$MODELS_DIR/qwen3-reranker-$SIZE"
GGUF="$WORK/model-f16.gguf"

say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

for c in llama-server curl find; do command -v "$c" >/dev/null || die "缺 $c"; done
python3 -c "import gguf" 2>/dev/null || die "缺 python 包 gguf。安装: pip install gguf(建议独立环境)"
if [ -f "$WORK/server.pid" ] && kill -0 "$(cat "$WORK/server.pid" 2>/dev/null)" 2>/dev/null; then
  SKIP_START=1
else
  SKIP_START=0
  lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "端口 $PORT 被其他进程占用"
fi

# ── 1. 官方权重自转(绝不用社区 GGUF——本书大坑) ──
if [ ! -f "$GGUF" ]; then
  say "下载官方权重 $HF_REPO"
  command -v hf >/dev/null || die "缺 hf CLI。安装: pip install -U huggingface_hub(建议独立环境)"
  hf download "$HF_REPO" --local-dir "$WORK/hf"
  say "新版 convert_hf_to_gguf.py 自转(带打分头)"
  CONVERT=$(find "$(dirname "$(command -v llama-server)")/.." -name "convert_hf_to_gguf.py" 2>/dev/null | head -1)
  # 多套 llama.cpp 并存时上面可能拿错;可用 CONVERT_PY=<path> 显式指定
  CONVERT="${CONVERT_PY:-$CONVERT}"
  [ -n "$CONVERT" ] || die "找不到 convert_hf_to_gguf.py,请从 llama.cpp 源码目录运行"
  python3 "$CONVERT" "$WORK/hf" --outfile "$GGUF" --outtype f16
fi

# ── 2. 打分头检查(30 秒救你于'伪装成模型弱'的深渊) ──
say "检查 cls.output.weight(缺=分数随机的坏文件)..."
python3 - "$GGUF" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
cls = [t.name for t in r.tensors if 'cls' in t.name]
assert cls, "FATAL: 无 cls 张量——打分头缺失,这个 GGUF 会输出随机分。用新版转换器重转。"
print(f"  打分头在位: {cls}")
PY

# ── 3. 启动+真实打分断言 ──
if [ "${SKIP_START:-0}" = 1 ]; then
  say "已有实例在跑(pid $(cat "$WORK/server.pid")),跳到断言(幂等)"
else
  say "启动 llama-server @ :$PORT"
  nohup llama-server -m "$GGUF" --reranking --pooling rank --ubatch-size 16384 --port "$PORT" > "$WORK/server.log" 2>&1 &
  SRV_PID=$!; echo "$SRV_PID" > "$WORK/server.pid"
  ok=0
  for i in $(seq 1 30); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 2; done
  if [ "$ok" != 1 ]; then kill "$SRV_PID" 2>/dev/null || true; rm -f "$WORK/server.pid"; die "60 秒未就绪(后台进程已清理),看 $WORK/server.log"; fi
fi
say "真实打分断言(相关文档应显著高于无关文档)..."
python3 - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
req = {"query": "什么是投机解码", "documents": ["投机解码用小模型起草再由大模型验证,加速生成。", "今天的午餐是意大利面。"]}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/rerank",
    json.dumps({"model": "m", **req}).encode(), {"Content-Type": "application/json"}), timeout=30)
scores = [d["relevance_score"] for d in sorted(json.load(r)["results"], key=lambda x: x["index"])]
print(f"  相关={scores[0]:.4f} 无关={scores[1]:.4f}")
assert scores[0] > scores[1], "FATAL: 相关文档分数不高于无关文档——打分链路异常"
assert scores[0] > 1e-10, "FATAL: 分数近零(4.5e-23 病),检查打分头"
PY
say "✅ 部署完成: http://127.0.0.1:$PORT/v1/rerank"
