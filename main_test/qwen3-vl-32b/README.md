# Qwen3-VL-32B-Instruct：MMTab-eval直接推理

原图加官方原题直接作答，不接入识表、子表、核查、返工或策略库。所有日志、结果和脚本均在本目录。

## 默认部署

机器2四张A800 80GB：GPU 0、1运行副本0（TP=2，端口8200），GPU 2、3运行副本1（TP=2，端口8201）。
内部端口起点分别54000和54200。两个副本依次初始化，避免同时加载压力；就绪后并行评测。
权重目录 `/nc1test1/lwj/Models/Qwen3-VL-32B-Instruct`，数据目录 `/nc1test1/lwj/data/MMTab_eval`。
每实例并发8，BF16，上下文32768，输出8192，temperature=0，seed=42，与8B保持评测口径。
显卡组、端口、上下文和调度参数都从config.json读取；可根据空闲卡调整gpu_groups，不能重叠。

**启动前先停止仍占用这四张卡的8B服务，并用nvidia-smi确认显存释放。** 不会自动结束其他实验进程。

```bash
cd /nc1test1/lwj/strategy/main_test/qwen3-vl-32b
# 已安装8B环境时复用；该步骤补齐tqdm和固定依赖
bash install.sh
bash test.sh check
bash start.sh
# 启动器逐个等待两个实例就绪；失败会打印日志路径
bash test.sh health
bash test.sh smoke
# 小批确认稳定后
bash test.sh full
```

默认解释器 `$HOME/.conda/envs/mmtab-qwen/bin/python`；覆盖方式：`PYTHON=/path/to/python bash start.sh`。
驱动与vLLM沿用8B已跑通的环境；32B双卡真实显存和吞吐尚未在机器2验证。
install.sh复用环境前应确保没有其他推理任务正在使用它。

## tqdm进度条与恢复

推理使用tqdm显示已完成/总数、速度与预计剩余时间，并显示本轮errors、truncated和per_min。
首次数据检查和图片哈希阶段先于推理进度条；尚未显示推理进度条不代表卡住。
errors和truncated为本次新增调用的计数，最终summary.json包含整个运行的统计。
已有记录会跳过，进度条从已保存数量开始，不重复调用。失败默认也跳过，需要明确重试：

```bash
bash test.sh full --retry-errors
# 修改参数必须另起目录，不能混合旧配置
bash test.sh smoke --run-name smoke_c4
```

不将执行失败移出分母，不自动重新生成空回答，不自动修改输出预算；length保留原回答评分。
只针对网络或部分服务错误做有限重试，可能有无法统计usage的超时调用，原始错误和尝试保留。
输出结构为outputs/smoke或outputs/full下的manifest.json、samples、predictions.jsonl、summary.json。
改变配置、代码或输入后使用新的run-name。异常退出遗留.lock时，确认无评测进程后再删除锁。

## 日志和停止

```bash
LOG=$(ls -t logs/server_8200_*.log | head -n 1)
tail -n 100 -f "$LOG"
# 单独恢复某一副本（先确认对应卡和端口已释放）
bash start.sh --replica 1
# 停止本目录管理的两个服务及其子进程
bash stop.sh
```

Ctrl+C退出tail不会停止服务。启动进程退出但遗留子进程时，stop脚本会提示检查，不会冒险终止未知PID。

## 离线评分

```bash
bash install_score.sh
bash score.sh
cat outputs/full/metrics/METRICS.md
# 已算完指标时，仅调整排列，不重算TEDS：
bash score.sh --report-only
```

沿用8B相同的冻结官方函数与TEDS转换、校验修复，不调用API。TQA/TFV按来源子集，结构任务按任务分别计分。
TR报告HTML TEDS、Markdown TEDS、LaTeX TEDS，各自以该格式全部样本为分母，转换失败记0。
BLEU为0–100，其余Accuracy/F1/TEDS为0–1，不混成一个总准确率。
RCE保留官方未解析目标计数局限；评分异常写入scoring_error。ToTTo不伪造缺失参考答案的本地分数。

结果按指定截图顺序固定输出：TQA的TABMWP、WTQ、HiTab、TAT-QA、FeTaQA、AIT-QA、TabMCQ；
TFV的TabFact、InfoTabs、PubHealthTab；T2T的ToTTo、HiTab_T2T、Rotowire、WikiBIO；
然后TSD行/列、TSD_OOD行/列、TCE、TCE_OOD、TCL、TCL_OOD、MCD、RCE行/列、RCE_OOD行/列、HTML/Markdown/LaTeX TEDS。
METRICS.md包含前后两张横向表。SUMMARY.tsv提供同序30项指标，另有TQA_TFV_T2T.tsv和TSU_TR.tsv便于分别复制。
展示表中的Acc/F1/TEDS为百分数并保留2位小数，BLEU保持0–100并保留6位小数；metrics.json原值不变。
缺失/不可用/评分异常显示“—”，不冒充0；真实零分显示0.00%。仅改变汇报方式，不改变8B已有的评分公式。

## 本地验证

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

模拟请求覆盖断点续跑、空回答、截断、配置冲突与参考答案隔离；评分验证覆盖换行校验、分母及三格式TEDS。


## 2026-11-26 评分更新

默认 `bash score.sh` 使用策略进化V2规范化口径，结果写入 `metrics_normalized/`。
原始口径使用 `bash score.sh --normalization raw`，仍写入 `metrics/`。
规则、输出和小批评分命令见 [NORMALIZATION.md](NORMALIZATION.md)。
