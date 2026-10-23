# Referee AI：多视角视频语义 + 规则 RAG + QLoRA

面向 SoccerNet-MVFouls 的可复现实验工程。系统先把多个机位的视频转换成“可观察事实”，再从指定赛季的 IFAB 规则库检索依据，最后输出判罚、逐条规则适用理由、引用和不确定性。

> SoccerNet 原始视频受其许可协议约束，仅限获授权的用途。本公开仓库只包含算法实现、配置模板和测试，不包含原始/派生视频、数据标注、事件清单、访问口令、模型权重或完整受版权保护的规则文本。

## 隐私与数据安全

- 请自行从官方渠道申请 SoccerNet 数据访问权限，并遵守对应许可与 NDA；不要在 Issue、日志或配置文件中提交访问口令。
- `data/raw/`、`data/metadata/`、`data/selected/`、`artifacts/`、`logs/`、模型权重及常见视频/压缩包格式已被 Git 与 Docker 构建上下文排除。
- `.env.example` 只提供变量名；真实令牌必须保存在未跟踪的 `.env` 或云平台密钥管理器中。
- IFAB 规则库应由使用者从获授权来源在本地构建；仓库仅定义最小 JSONL 数据契约。
- 发布派生统计前，应检查其是否可能还原比赛、球员、时间戳或受限标注。

## 核心设计

```text
多视角视频 → 单视角事实 JSON → 跨视角证据融合 → 混合规则检索
                                               ↓
标签/人工事实 → QLoRA训练             引用约束判罚 + 解释 + 弃权
                                               ↓
                              分类指标 + 校准 + 引用/一致性指标
```

本项目的创新性主要体现在三个可验证机制：

1. **Evidence Agreement Calibration（证据一致性校准）**：每个事实保留来源机位、时间戳和置信度；机位冲突不会被静默平均，而会降低最终置信度并触发 `cannot_determine`。
2. **Rule-grounded Structured Verdict（规则约束结构化判罚）**：事实、规则和结论是三个独立层；每个规则应用必须绑定检索到的 chunk ID，避免生成“看似专业但无依据”的解释。
3. **Counterfactual Consistency（反事实一致性）**：实验会改变“触球、鞋钉、力度、接触点”等单一事实，检查判罚是否按规则方向稳定变化；另设乱序规则和无视频对照，验证收益确实来自视频与RAG。

## 数据切分

- train：2916 个事件，用于 QLoRA。
- valid：411 个事件，用于选择 checkpoint、提示词和阈值。
- test：301 个事件，只在所有设计冻结后做一次最终评估。

禁止使用当前已下载的 301 个 test 事件训练或选择超参数。多个视角属于同一事件，必须始终处于同一 split。

## 租卡建议

默认配置是一张 **L40S 48GB**；预算有限可用 RTX 4090 24GB 并按 [rental_gpu.yaml](configs/rental_gpu.yaml) 降低帧数和上下文；A100 80GB 更稳。正式模型为 `Qwen/Qwen3-VL-8B-Instruct`，采用 NF4 QLoRA、BF16 和梯度检查点。

```bash
git clone https://github.com/haolin808li-png/soccer-referee-rag.git referee-ai
cd referee-ai
bash scripts/cloud_bootstrap.sh
```

若判罚模型使用同一张卡上的 OpenAI-compatible 服务，可在完成视频语义或训练后启动：

```bash
python -m pip install -e '.[serve]'
bash scripts/serve_decision_model.sh
```

受限数据只能通过私有、加密方式传到临时云盘。训练结束下载 `artifacts/runs` 后销毁云盘；不要制作含原视频的公开镜像或公开数据卷。

## 仓库边界

公开仓库提供以下内容：

- 多视角抽帧、结构化语义生成与证据融合；
- BM25/向量混合检索、引用约束判罚与拒答机制；
- QLoRA 数据构造、训练入口、评估指标与反事实测试；
- CPU dry-run、单元测试和租用 GPU 的配置模板。

公开仓库不提供 SoccerNet 数据下载口令、数据副本、预生成事件清单、模型 checkpoint 或实验输出。脚本只会读取使用者在本地准备的数据。

## 快速验证

```bash
python -m pip install -e '.[dev]'
make test
make dry-run
make preflight
```

`dry-run` 仅检查流水线命令，不下载模型。各命令均支持 mock/dry-run 路径，CPU 环境也能完成单元测试。
`preflight` 会核对官方事件数、全部视频路径、切分泄漏、规则版本、规则索引与 FFmpeg；上云后增加 `--require-cuda` 验证GPU。

## 正式流程

先下载并审计官方 train/valid；test 已有清单可继续使用。随后依次执行：

```bash
# 1. 每个 split 生成可缓存语义；中断后可恢复
python scripts/run_pipeline.py --config configs/pipeline.yaml --split train --stages semantics
python scripts/run_pipeline.py --config configs/pipeline.yaml --split valid --stages semantics

# 2. 将你有权使用的 IFAB 规则文本整理为 JSONL 后建索引
python scripts/run_pipeline.py --config configs/pipeline.yaml --stages index

# 3. 生成带引用的判罚
python scripts/run_pipeline.py --config configs/pipeline.yaml --split valid --stages decisions

# 4. 构造训练数据并训练（绝不读 test）
python scripts/run_pipeline.py --config configs/pipeline.yaml --stages prepare train

# 5. 冻结方案后进行最终测试
python scripts/run_pipeline.py --config configs/pipeline.yaml --split test --stages evaluate
```

也可用统一入口：

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml \
  --stages semantics index decisions prepare train evaluate
```

最终 `evaluate` 被代码强制限定为 `--split test`。生成训练语义时请分别以
`--split train` 和 `--split valid` 调用，禁止在训练阶段使用test标签调参。

## 规则数据格式

每行一个最小、可引用的规则单元：

```json
{"chunk_id":"ifab-2025-law12-dfk-careless","version":"2025/26","law":"12","section":"Direct free kick","concept":"careless","sanction":"direct_free_kick","text":"<你有权使用的规则摘要或文本>","source_url":"<官方来源>"}
```

不要混用不同赛季规则。规则内容被当作不可信数据，不允许其中的文字覆盖系统指令。

## 产物契约

每个正式训练运行目录必须包含：

```text
artifacts/runs/<run_id>/
├── config.yaml
├── splits.csv
├── metrics.json
├── predictions.csv
├── models/
└── reports/report.md
```

除 offence、severity、card 的 macro-F1 / balanced accuracy 外，还评估 ordinal error、ECE、弃权覆盖率、引用命中率、规则忠实度和反事实一致性。完整实验矩阵见 [pipeline.yaml](configs/pipeline.yaml)。
