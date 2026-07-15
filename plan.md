# LIGM 两阶段研究计划

## 研究目标与方法

目标是在 `answerdotai/ModernBERT-base` 上验证：

> 根据正常全局注意力与全局层局部化后的预测差异选择 MLM 目标，能否强化长程依赖，并提高 MLDR-English OOD 单向量检索。

对 40% whole-word 候选位置，用 EMA 教师执行正常注意力 \(G\) 与 all-local 反事实 \(L\)：

\[
g_i=\log p_G(x_i)-\log p_L(x_i)
\]

\[
s_i=\max(g_i,0)\cdot4p_G(x_i)(1-p_G(x_i))
\]

最终训练位置保持 30%：

- 20%：最高分 span。
- 10%：剩余候选中的随机 replay。
- 学生使用标准 MLM 交叉熵。
- EMA decay 固定为 `0.999`。
- 不加入 KD、对比损失、LoRA 或架构修改。

反事实前向临时将全局层的 `local_attention=(-1,-1)` 切换为与局部层相同的窗口，层内已构造的 global RoPE 和所有权重保持不变。[Transformers 4.57.6 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py)

## 下载、环境与数据

### uv 环境

台式机使用独立 Python 3.11 环境，所有依赖写入 `pyproject.toml` 和 `uv.lock`：

```text
uv python install 3.11
uv venv --python 3.11
uv lock
uv sync --frozen
```

`uv` 当前尚未安装，先安装官方单文件版本。项目默认 index 固定为：

```toml
[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true

[tool.uv]
find-links = ["https://mirrors.aliyun.com/pytorch-wheels/cu124/"]
```

使用 `uv lock` 生成完整锁文件，训练时只允许 `uv sync --frozen`，不临时升级依赖。该配置遵循 uv 官方的自定义 index 和锁文件机制。[uv 文档](https://docs.astral.sh/uv/)

### Hugging Face 镜像

固定设置：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/nvme-data/ligm/hf
HF_HUB_CACHE=/nvme-data/ligm/hf/hub
```

模型与数据下载完成后，训练设置 `HF_HUB_OFFLINE=1`，只从本地读取。`hf-mirror.com` 已在台式机上验证可访问；不在失败时静默切回 Hugging Face 官方源。[HF-Mirror](https://hf-mirror.com/)

下载流程：

1. 通过镜像 API 解析并固定仓库 commit。
2. 生成包含文件 URL、输出路径、大小和校验值的 aria2 manifest。
3. 模型等少量大文件使用：
   - `aria2c -j4 -x16 -s16 -k1M -c`
4. MDS 数据 shard 使用：
   - `aria2c -j16 -x4 -s4 -k1M -c`
5. 固定 `--auto-file-renaming=false` 和 `--file-allocation=none`。
6. 下载结束后校验 SHA256/ETag；缺失或不匹配直接报错。
7. 不直接执行远程下载脚本，aria2 manifest 由项目代码生成。

Hugging Face Hub 自身也支持自定义 endpoint、local cache 和并发下载，但本项目只用它解析仓库元数据，实际大文件交给 aria2c。[Hub 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)

### 数据

数据来自 Ettin extension/decay 的长文档 MDS shard：

- 30% books/textbooks
- 25% arXiv/PeS2o
- 20% DCLM Dolmino
- 10% Wikipedia
- 10% StackExchange
- 5% code

只从单一文档内部裁剪序列，不拼接无关文档。按文档 ID 稳定哈希划分 98%/1%/1% train/validation/test。[Ettin extension](https://huggingface.co/datasets/jhu-clsp/ettin-extension-data)、[Ettin decay](https://huggingface.co/datasets/jhu-clsp/ettin-decay-data)

目录统一放在 `/nvme-data/ligm`；数据和下载缓存最多使用 330GB，checkpoint 最多 100GB，始终保留至少 40GB 空间。

## 第一阶段：验证假设

### 训练范围

建立约 1B-token 的固定数据快照，但每个实验只训练 100M token，种子固定为 `11`。

首先只运行：

- 随机 30% MLM。
- 完整 LIGM。

配置固定为：

- BF16
- 长度比例：50% × 2048、30% × 4096、20% × 8192
- micro-batch：4/2/1
- gradient accumulation：4
- gradient checkpointing：开启
- StableAdamW
- learning rate：`2e-5`
- weight decay：`0.01`
- 2% warmup、83% stable、15% decay

通过初步机制门槛后，再补充两个 100M-token 对照：

- Entropy-aware masking：排除普通难度选择的作用。[相关论文](https://aclanthology.org/2026.starsem-conference.27/)
- LIGM 仅使用 \(g_i\)：检验 learnability 项。

### 必要评测

只保留三类评测：

1. 合成长依赖恢复：
   - 128–512
   - 512–2048
   - 2048–4096
   - 4096–8192 token

2. 固定自然文档 MLM：
   - 0–128 token 局部恢复
   - 128 token 以上长程恢复

3. 廉价 MLDR probe：
   - 原始 ModernBERT、随机 MLM、完整 LIGM。
   - 只使用固定 250K MS MARCO hard-negative 子集。
   - 一个检索种子。
   - 只评 MLDR-English dev OOD。
   - Entropy masking 仅在机制指标距离 LIGM 不超过 0.2 点时补做该 probe。

不运行 GLUE、BEIR、MLDR-ID、代码检索、ColBERT 或完整超参数 sweep。

### 进入第二阶段的门槛

必须全部满足：

- LIGM score 与已知支持距离的 Spearman 相关系数至少 `0.2`。
- 512 token 以上恢复准确率相对随机 MLM 提高至少 5%。
- 0–128 token 恢复准确率绝对下降不超过 `0.5` 点。
- MLDR dev nDCG@10 相对随机 MLM 至少提高 `0.5`。
- LIGM 至少不弱于 Entropy-aware masking。

任一核心门槛失败即停止扩大训练，记录负结果，不通过增加 token 数碰运气。

## 第二阶段：放开训练

### 正式训练

仅正式扩展：

- 完整 LIGM：种子 `11/22/33`。
- 随机 30% MLM token-matched：种子 `11/22/33`。
- 随机 MLM compute-matched：只运行种子 `11`，训练到与 LIGM 相同 GPU 小时。
- Entropy masking：仅当第一阶段 MLDR probe 距 LIGM不超过 0.5 nDCG 时运行一个正式种子。

训练以 500M token 为一个 block，不预设总 token 上限：

- 每 500M token 运行合成与自然 MLM 机制评测。
- 每 1B token 仅对种子 11 运行一次 250K MS MARCO → MLDR dev probe。
- 种子 11 决定停止预算，其他种子随后训练到完全相同的 token 数。
- Checkpoint 每 250M token 保存一次，只保留最近两个、每个 1B 边界和最终候选。

满足以下任一条件即停止继续扩展：

- 连续两个 1B-token probe 的 MLDR dev 提升均小于 `0.2`，且长程 MLM 提升均小于 1%。
- 0–128 token 恢复相对随机 MLM下降超过 `0.5` 点。
- LIGM 与随机 MLM 的差距连续两个 probe 缩小。

最终 checkpoint 按 MLDR dev 选择；MLDR test 在模型和统计方案冻结后只运行一次。

### 最终下游评测

只保留一个完整下游任务：

- 使用完整 1.25M MS MARCO hard-negative 数据训练单向量检索器。
- 有效 batch 16，5% warmup。
- 原始 ModernBERT、随机 MLM、LIGM 各运行三个配对种子。
- 只报告 MLDR-English OOD test nDCG@10。
- 使用查询和种子的分层配对 bootstrap，10,000 次重采样。

最终成功条件：

- LIGM 相对重新复现的原始 ModernBERT 至少 `+1.0 nDCG@10`。
- LIGM 相对同 token 随机 MLM 至少 `+1.0 nDCG@10`。
- 两个差值的 95% 置信区间下界均高于 0。
- 0–128 token 自然 MLM 不下降超过 `0.5` 点。
- 单独报告 compute-matched 随机 MLM；若 LIGM 只超过 token-matched 对照，论文结论限定为 token efficiency。

## 验证与交付

实现前必须通过：

- 8K batch 1 在 RTX 3090 上稳定前向、反向和保存 checkpoint。
- 短于局部窗口时 G/L 输出一致。
- all-local 只改变 attention mask，不改变 RoPE 或权重。
- whole-word mask 比例严格为 20% LIGM + 10% replay。
- EMA 无梯度且中断恢复后数据顺序、mask 和 loss 可复现。
- 所有方法读取相同的数据 manifest 和 crop 顺序。
- aria2 下载产物通过 revision、大小和哈希校验。

最终公开 GitHub 代码、`uv.lock`、数据 manifest、配置、原始指标、统计脚本及 Hugging Face checkpoint。论文结论限定为 ModernBERT-base 的长程 MLM 与 MLDR-English OOD 单向量检索。
