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
- 长度比例：50% × 2048、30% × 4096、20% × 8000（Ettin MDS 原生样本长度）
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
   - 512 token 以上长程恢复

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

### 第一阶段结果与前瞻性修订

第一阶段于 2026-07-16 完成。随机 MLM 与完整 LIGM 均使用种子 `11` 训练
`100,006,238` token。LIGM 的固定自然文档长程恢复相对随机 MLM 从
`48.624%` 提高到 `49.043%`，绝对提高 `0.419` 点、相对提高 `0.862%`；
局部恢复从 `84.006%` 变为 `83.810%`，绝对下降 `0.196` 点。合成四桶平均
恢复准确率从 `13.281%` 提高到 `26.562%`，但 LIGM score 与距离的
Spearman 相关系数为 `-0.8`。

因此，原第一阶段门槛失败，Entropy、仅使用 \(g_i\) 和 MLDR probe 均未运行。
该结论保持不变。以下第二阶段是看到第一阶段结果后制定的探索性追加实验，不能
用于声称原门槛已经通过，也不能覆盖第一阶段的负结果。

## 第二阶段：放开训练

### 在线评测与硬停止

第二阶段将评测嵌入训练控制流程，而不是等训练结束后再决定：

1. 先将种子 `11` 的随机 30% MLM 从现有 100M checkpoint 续训至最高 1B
   token，建立 token-matched 参考曲线；续训前先在现有随机与 LIGM 100M
   checkpoint 上用新的 128 篇固定验证集建立共同起点。
2. 每 25M token 暂停一次训练，保存 checkpoint，并在固定的 128 篇验证文档
   上运行自然重复恢复评测。
3. 保存逐文档、局部桶和长程桶的原始计数与准确率；同一评测集、mask seed 和
   数据顺序在所有方法及所有检查点之间保持不变。
4. 随后续训种子 `11` 的 LIGM。每个检查点与相同 token 数的随机 MLM 结果
   配对，定义：

   \[
   \Delta_{local}(t)=Acc^{LIGM}_{local}(t)-Acc^{Random}_{local}(t)
   \]

5. 若任一在线评测满足 \(\Delta_{local}(t)<-0.005\)，立即正常终止当前 LIGM
   run，不继续到下一个 token 区间，并选择上一个满足门槛的 checkpoint 作为
   安全候选。该判断不要求连续两次越界。
6. 自然长程恢复随每个 25M checkpoint 记录；合成四桶准确率与
   information-gain 统计只在 `100M / 250M / 500M / 750M / 1B` 里程碑运行。
   这些指标均不覆盖 `-0.5` 点局部恢复硬停止规则。

在线评测通过单卡顺序执行：训练到评测点后释放训练临时张量，在同一张 RTX 3090
上完成评测，再恢复训练。不得同时运行训练和评测进程争用显存。

### Checkpoint 与预算

- Checkpoint 和自然在线评测间隔均为 25M token，约对应当前配置下的
  980–1000 个 optimizer step；以 token 数而不是 step 数作为唯一触发条件。
- 最高训练预算固定为 1B token，不在第二阶段内继续提高上限。
- 永久保留 `100M / 250M / 500M / 750M / 1B` checkpoint，并始终保留最近
  两个 25M checkpoint，确保触发硬停止后可以回到上一个安全点。
- 每次评测的 JSON 和逐文档记录永久保留，不随 checkpoint 清理。
- 250M 和 500M 是趋势检查点；只有此前未触发局部硬停止才继续。1B 是最终上限，
  不是必须消耗完的目标。

### 对照与种子晋级

- 种子 `11` 先运行完整的 token-matched 随机曲线，再运行带在线硬停止的 LIGM。
- 随机 MLM compute-matched 只运行种子 `11`，其 GPU 时间与最终入选 LIGM
  checkpoint 相同。
- 种子 `22/33` 先各运行 100M 的随机 MLM 与 LIGM，确认长程差值方向是否一致。
- 只有种子 `11` 在 500M 仍保持长程优势且未触发局部硬停止，种子 `22/33`
  才晋级到相同 token 预算；所有晋级 run 使用同一在线硬停止规则。
- Entropy masking 和仅使用 \(g_i\) 的消融不因扩大 token 预算自动解锁；只有主实验
  在 500M 仍保持长程优势后，各运行一个种子和相同 token 预算。

### 检索评测解锁

- 250M 只运行机制评测，不运行检索任务。
- 种子 `11` 在 500M 仍保持自然长程恢复优势且未触发局部硬停止时，解锁一次固定
  250K MS MARCO → MLDR-English dev probe。
- 若继续至 1B，在 1B 再运行一次相同 probe。
- 最终 checkpoint 只能从通过局部硬停止门槛的候选中按 MLDR dev 选择；MLDR
  test 在模型、统计方案和 checkpoint 全部冻结后只运行一次。

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

### LIGM v2：`+1.0` 长程恢复筛选

完整 1B 配对曲线结束后，不从主 LIGM checkpoint 继续堆 token。以随机 MLM 的
1B checkpoint 作为新的共同起点，先运行一次 25M token、仅含 8K 序列的短筛选：

1. `random-long`：30% whole-word random MLM，作为相同数据、序列长度和 token
   预算的基线。
2. `weighted4`：先采样与训练输入完全一致的 30% 掩码，在这份输入上计算
   global-vs-all-local information gain；最高分的 20% token 使用 4 倍损失权重。
3. `weighted8`：与 `weighted4` 相同，但定向 token 使用 8 倍损失权重，用于判断
   当前方法是否主要受长程梯度稀释限制。

三个分支重置优化器并使用独立的 25M warmup-stable-decay 调度，数据位置、随机数
状态和初始模型完全一致。每个分支结束后在固定 128 篇文档上配对评测，并继续执行
局部恢复不低于 `-0.5` 点的硬门槛。

短筛选的晋级规则为：长程恢复相对 `random-long` 至少 `+0.6` 点，且逐文档配对
bootstrap 的 95% 区间下界高于 0。只有满足该规则的权重才扩展到 100M；100M
目标为长程恢复 `+1.0` 点。若两个权重均未达到 `+0.6`，不扩大 token 预算，转而
判定现有 token 选择目标不足以产生 1 点收益。

### RED-LIGM：远程证据删除课程

加权筛选未达到晋级线后，不再提高 LIGM 权重。新假设是 all-local 反事实混入了
注意力架构变化，而没有隔离具体远程证据。RED-LIGM 在同一份 30% 随机掩码上
建立完整视图和证据删除视图：对没有 128-token 内同词证据、但存在至少 512-token
外未掩码同词证据的目标，删除其远程出现位置，使用两个视图的真实 token 对数概率
差作为分数。最高分的 10% token 使用 2 倍损失权重，其余 20% 保持普通随机 MLM。

第一阶段从随机 MLM 1B checkpoint 分叉三个完全配对的 10M run：`random`、
`red-full` 和 `red-route`。`red-route` 只更新编号能被
`global_attn_every_n_layers` 整除的全局注意力 block，冻结嵌入、MLM head 和局部
block，用于检验局部退化是否来自参数冲突。每 5M checkpoint 和评测；10M 长程
提升达到 `+0.4` 点才扩展至 25M，25M 达到 `+0.7` 点且局部下降不超过 `0.25`
点才进入第二阶段。

第二阶段只保留胜出配置，从共同起点重新训练 100M。每 10M 评测，两个相邻点达到
`+0.8` 后才运行锁定的 512 篇测试集；最终目标为测试集长程恢复 `+1.0` 点、局部
下降不超过 `0.5` 点。只有达到该门槛才解锁 MLDR probe 和额外种子。

### NA-RED：局部锚定的零空间更新

RED-full 在 10M 达到 `+0.679` 点长程提升但局部下降 `0.378` 点；只更新全局
block 的 RED-route 将局部下降减至 `0.263` 点，同时长程提升降至 `+0.506`。
NA-RED 保留全模型更新，把随机 MLM 梯度作为基线，并使用冻结的共同起点教师在
局部证据 token 上进行 top-32 蒸馏。RED 额外梯度与局部保护梯度点积为负时，删除
冲突分量；投影后的远程梯度按基线与远程梯度范数比自动缩放到 `[0.5, 2.0]`。

先从共同起点运行 10M，复用 stage4 的随机曲线。5M 要求长程至少 `+0.4` 点且局部
下降不超过 `0.15` 点；10M 要求长程至少 `+0.6` 点且局部下降不超过 `0.2` 点。
只有满足两项才扩展至 25M。

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
