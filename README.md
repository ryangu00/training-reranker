# Reranker 自托管与训练:一个把"文件坏"伪装成"模型弱"的社区级大坑

> 检索三支柱:embedding(召回)→ **reranker(精排)** → 生成。这本书讲 Qwen3-Reranker 的自托管部署、
> 我们的训练集管线,和一个值得所有人知道的坑:**我们当时(2026-07)下载的多个社区转换 reranker GGUF 均缺打分头,分数近乎随机,而且不报任何错。**

## 大坑先行:GGUF 缺 `cls.output.weight`

**症状**:llama.cpp `--reranking` 模式跑起来一切正常,但所有文档得分都是 4.5e-23 这种近零值;接进检索链路后精排结果**比不用还差**(我们域内留出集实测 recall@1 从 0.5667 掉到 0.0417——低于随机:候选 k 里随机挑的期望 ≈1/k,它是自信地把对的排下去)。
**根因**:这些 GGUF 缺 `cls.output.weight`(打分头张量)——我们当时所用时期的转换工具不认这个头,静默丢弃(后续版本已修,故必须用新版自转)。没有头,`--reranking` 拿不到打分投影,输出垃圾值,**不报错**。
**诊断方法**(30 秒):
```bash
python3 -c "
from gguf import GGUFReader
r = GGUFReader('<model.gguf>')
print([t.name for t in r.tensors if 'cls' in t.name])"
# 空列表 = 坏文件;应看到 cls.output.weight 等
```
**修法**:不用社区 GGUF,从官方 HF 权重(如 `Qwen/Qwen3-Reranker-0.6B`/`-4B`)用**新版** llama.cpp 的 `convert_hf_to_gguf.py` 自转——带头的产物 tensor 数明显不同(当时版本 0.6B:311 个 / 4B:399 个,可当校验指纹;不同版本数字或异,核心是 cls 张量必须在)。

## 一键部署

`scripts/deploy.sh --size 0.6B|4B`——下载官方权重→新版自转→**cls.output.weight 自动检查**(本书大坑的机器化)→启动→真实打分断言(相关>无关且非近零)。

## 部署(llama.cpp server)

```bash
llama-server -m <自转带头的.gguf> --reranking --pooling rank --ubatch-size 16384
```
- `--ubatch-size 16384`:预防性设置——保证长文档对(数千 token 级)不被 batch 边界截断(同 embedding 篇 num_batch 的教训)。
- 接线注意:上游读的是配置里的 rerank 端点字段——**打一发真实请求断言分数分布正常**(好模型的分数应有明显区分度,不是全部挤在 0 或 1 附近),别只验服务 200。

## 效果口径(我们的三方对比)

| 方案 | 检索质量增益(我们的留出集) |
|---|---|
| 无 reranker(纯 embedding 召回) | baseline |
| +Qwen3-Reranker-0.6B(自转带头) | 约 +4 个点 |
| +Qwen3-Reranker-4B(自转带头) | 约 +11 个点 |

4B 增益显著更高;0.6B 胜在延迟。按你的延迟预算选。(口径说明:我们域内留出集的内部综合检索分——绝对值不可跨环境比较,**方向与相对量级**是可迁移的参考。)

## 训练(如需进一步特化)

reranker 训练集 schema 与 embedding 篇同源管线生成(query + 正/负段落对),规模需求比 embedding 大(MB 级 JSONL 量级)。实践建议:**先用官方权重+自转修头,确认精排链路健康后再谈训练**——我们的最大收益来自"把坏文件换成好文件"(0.0417→0.5667+),训练特化是那之后的增量。训练数据不随 repo 发布(含私有语料),schema 与生成脚本同 embedding 篇。

## 可复用结论

- **"伪装成模型弱"的文件级故障是最危险的一类**:不崩、不报错、指标就是差。对任何下载的量化产物,先做张量清单检查(缺头/缺层),再做行为断言(分数分布/已知样例),最后才评"模型好不好"。
- 精排链路上线断言三件套:张量指纹 ✓ / 真实请求分数分布 ✓ / 留出集对比 baseline ✓。

---
*RyanAI Lab · 一手事故与三方对比实测,更新于 2026-09。欢迎 issue 反馈。*
