# Qwen3-VL-8B-Instruct：MMTab-eval 直接推理

本目录独立保存模型测试相关脚本、配置、服务日志和输出。权重与数据保留在共享目录，不复制进本目录。
只发送原图和官方 `input` 原文，无识表、子表、策略、核查或答案修订。

## 机器2运行

将整个 `main_test/qwen3-vl-8b` 文件夹上传到机器2的 strategy/main_test 下；不要求 strategy 位于某个绝对路径。
进入上传后的 qwen3-vl-8b 目录执行以下命令。脚本始终以自身所在目录为基准。

```bash
# 环境尚未安装时执行；安装完成的机器可跳过
bash install.sh

# 在不调用模型的情况下检查全部标注、图片唯一性和图片哈希
bash test.sh check

# 空闲卡编号不是0,1,2,3时修改GPUS；按顺序对应8100到8103端口
GPUS=0,1,2,3 bash start.sh

# 启动编译完成后检查四个服务是否就绪
bash test.sh health

# 每个 task_type + dataset_name 组合固定随机抽4题，四服务各并发8
bash test.sh smoke

# 小批结果核验且无超长/显存问题后，再启动全量
bash test.sh full
```

`health` 尚未就绪会报错，此时查看 logs 下对应端口日志；不重复运行 start.sh。
安装采用vLLM 0.11.0默认CUDA 12.8包，截图550驱动不保证支持全部编译路径。
Transformers固定为4.57.1，避免新版本移除all_special_tokens_extended导致旧vLLM启动失败。
start.sh会先在CPU上检查本地tokenizer与vLLM兼容性，通过后才启动四个服务。
若出现driver/PTX错误，保留日志并适配驱动或安装方案；仅pip安装成功不代表服务可用。
安装和真实GPU推理尚未在机器2验证。

## 默认路径

- 权重：`/nc1test1/lwj/Models/Qwen3-VL-8B-Instruct`
- 标注：`/nc1test1/lwj/data/MMTab_eval/MMTab-eval_test_data_49K.json`
- 图片：递归扫描 `/nc1test1/lwj/data/MMTab_eval/images`，自动兼容 `images/all_test_image` 等层级。
- 配置：本目录 `config.json`；端口与start.sh对应，改端口时需同步两处。
- 环境：`$HOME/.conda/envs/mmtab-qwen`；自定义环境时使用 `PYTHON=/path/to/python bash start.sh` 或 `bash test.sh ...`。

当前启动参数：BF16、每卡一副本、上下文32768、输出上限8192。图片处理沿用权重中的processor配置，不人为降低分辨率。
上下文由16384扩大到32768是根据机器2小批测试中16489-token输入及仅剩11-token生成空间的记录修正。
`server_max_model_len`在config中用于实验记录；实际启动限制在start.sh/start_one.sh中，修改时必须同步。
新设置应在新输出目录重新评测，不能续写旧smoke结果。7条8192-token长输出尚需检查重复或证据不足，不能仅凭length断定输出预算不足。
上下文不足会记录请求错误；达到输出上限会标记length，不静默截短输入或自动加长后重答。
配置和脚本版本、标注哈希、每张图片哈希、固定样本清单写入manifest，参考答案不进入请求。

## 输出与恢复

```text
qwen3-vl-8b/
  logs/                     # 四服务日志和环境记录
  pids/                     # 服务PID
  outputs/
    smoke/                  # 默认小批目录
    full/                   # 默认全量目录
      manifest.json         # 冻结输入和配置
      samples/*.json        # 每题原始响应、答案、usage、耗时、错误和请求尝试
      predictions.jsonl     # question_id/text，与官方预测格式一致；失败保留为空答案
      summary.json          # 完成数、错误数、截断数、token用量和吞吐
      previous_attempts/    # 显式重试错误前保留旧记录
```

重跑相同命令自动跳过已有记录，包括失败记录。明确重试失败使用：

```bash
bash test.sh smoke --retry-errors
bash test.sh full --retry-errors
```

仅429、特定5xx和连接/超时失败自动重试；内容错误和空回答不自动重新采样。
超时重试可能在服务端产生额外计算，无法取得usage的调用不伪造token用量。
输出上限、并发、输入等修改后必须使用新目录，示例：

```bash
bash test.sh smoke --run-name smoke_c4 --per-group 4
```

修改配置后旧目录会拒绝续跑，避免混合不同实验设置。结束进程异常留下.lock时，先确认该评测进程已退出再手动删除锁文件。
运行汇总不是准确率。评分入口见下节；沿用冻结的官方notebook函数和项目已有TEDS转换方式，不引入模型裁判或答案重写。
小批选题参考任务分组只用于抽样，不作为模型输入。

## 停止和本地验证

每个副本使用不同的VLLM_PORT内部通信端口起点（52000、52100、52200、52300），与HTTP接口端口不同。
如果只有部分副本失败，确认对应显卡没有遗留进程、HTTP端口未监听后，可单独恢复：

```bash
bash start_one.sh 1 8101
bash start_one.sh 3 8103
```

内部端口仍可能被其他程序占用；脚本检查HTTP和内部起始端口，实际启动是否成功以日志和health为准。

```bash
# 只停止本目录PID与端口匹配的vLLM服务
bash stop.sh

# 无需模型/GPU，模拟HTTP接口验证并发请求、原题输入、断点续跑、截断与配置冲突
python -m unittest discover -s . -p test_evaluate.py -v
```

首次编译/模型加载不计入推理计时；首轮推理仍可能存在预热成本。
吞吐数字包括客户端图像读取和请求时间；单题seconds包含传输重试，不能等同GPU纯计算时间。

## 离线评分（2026-11-23新增）

将score.py、score.sh、install_score.sh、score_requirements.txt及整个scoring_source目录同步到机器2同名目录。
无需重新跑推理，不需要启动vLLM，不调用模型API，不占GPU。评分环境单独安装，避免改动推理依赖。

```bash
bash install_score.sh
bash score.sh
# 默认评分 outputs/full；其他运行目录示例：
bash score.sh --run outputs/smoke_ctx32768
# TEDS默认4个CPU进程，可指定：
bash score.sh --workers 8
```

结果位于对应运行目录的metrics/METRICS.md和metrics/metrics.json，TEDS逐题缓存位于metrics/teds_cache。
TR较慢，可以重跑同一评分命令复用缓存。预测读取samples原始记录，不依赖可能过期的predictions.jsonl。
缺少任何选定样本会拒绝评分；执行失败的样本仍提供空答案，截断回答原样评分。
全量运行会额外检查样本ID集合与全部官方标注相同，且核验标注哈希。

指标：TQA/TFV按来源子集Accuracy，FeTaQA/T2T按来源corpus BLEU，TSD行列Accuracy，TCE/TCL单元格Accuracy，MCD F1，RCE行列P/R/F1，TR TEDS。
评分notebook校验统一CRLF/LF换行后计算哈希，允许Git跨平台换行转换，仍拒绝其他内容修改。
Accuracy/F1/TEDS为0–1，BLEU为0–100，不混算总准确率。来源代码和依赖版本记录到metrics.json。
ToTTo官方测试参考缺失，不伪造本地指标。TR沿用项目已有转换，并以全部选定样本为分母，转换/计算失败记0。
TR按原题要求分别汇总HTML_TEDS、Markdown_TEDS、LaTeX_TEDS；各自以该格式全部样本为分母，缺少该格式时显示“—”。
格式由标注original_query_type决定，不由模型实际回答格式决定；整体TEDS作为JSON辅助字段保留，不替代三列。
其余官方函数保留原公式，包括RCE对未解析目标的计数局限；分母不是统一的整题数。
官方函数异常会写入scoring_error并明确标记，不能把空指标当0或声称全部任务已成功评分。
