# Table-LLaVA-7B 原始版 MMTab-eval

模型为 `/nc1test1/lwj/Models/table-llava-v1.5-7b`，配置中架构为LlavaLlamaForCausalLM。
不是llava-v1.6-vicuna-7b-hf，也不使用SynTab。CLIP目录默认 `/nc1test1/lwj/Models/clip-vit-large-patch14-336`。
需要本地完整CLIP权重、config.json及preprocessor_config.json；不存在时先下载CLIP，不能用其他视觉编码器代替。

## 安装与四卡小批测试

先停止占用GPU 0、1、2、3的Qwen服务，确认显存释放。本目录不自动结束其他实验。
不用运行start.sh或health，没有HTTP服务；一个进程加载一次原始7B模型，负责一张卡上的分片。

```bash
cd /nc1test1/lwj/strategy/main_test/table-llava-7b
bash install.sh
bash test.sh check
bash test.sh smoke
# 检查完小批结果后再执行
bash test.sh full
```

独立环境 `$HOME/.conda/envs/mmtab-table-llava`，Python3.10及作者pyproject中指定的旧依赖。
Qwen和评分环境不改变。代码克隆到本目录vendor/Table-LLaVA，安装时记录commit及pip freeze。
仓库首次克隆需要联网，推理强制使用本地权重，不在后台下载CLIP。

原始权重保持不变：prepared_model内为权重符号链接，仅复制config并将mm_vision_tower指向本地CLIP。
使用作者load_pretrained_model、llava_v1聊天模板、image_processor.preprocess及贪心生成。
预处理遵循Table-LLaVA原生CLIP配置，不强行改成Qwen的动态分辨率。

官方参考：https://github.com/SpursGoZmy/Table-LLaVA/blob/main/llava/eval/model_vqa.py
默认输出上限1024（作者该脚本设置），**不是Qwen的8192**；模型原生上下文可能更短。
实际生成预算为min(1024,原生上下文减图文输入)，requested_max_tokens/effective_max_tokens/context_limit均逐题保存。
输入已超过原生上下文时记失败，不截断原题、缩图重试或改答案。
若要研究统一输出预算，需要先验证模型能力后另起实验，不冒充同设置对照。

## 输出与恢复

默认gpu_ids为0、1、2、3，可在config.json改为空闲卡编号。四条tqdm分别报告各GPU的分片进度。
同Qwen按task_type和dataset_name固定seed=42抽样，每组4条，默认同一批368条。
outputs/smoke及outputs/full保留manifest、逐题samples、predictions.jsonl及summary.json。
生成token IDs、原始答案、token用量、耗时及失败原因写入每题JSON，length仅标记不重采样。
相同命令续跑会跳过已有记录；失败也保留，需明确重试：

```bash
bash test.sh smoke --retry-errors
bash test.sh smoke --run-name smoke_v2
```

上下文预算缩减数见summary.context_limited。模型加载失败会使整轮报错，缺题不会伪装完成。
进程崩溃遗留.lock时，先确认旧任务已退出，再删除对应锁文件。
数据匹配保留.html.csv等完整ID；标注哈希、图片哈希和官方代码commit记录于manifest。
模型参数文件使用原始路径引用，开始正式实验后勿改动权重；manifest校验模型config，未哈希全部权重分片。

## 评分

```bash
bash install_score.sh
bash score.sh
cat outputs/full/metrics/METRICS.md
# 小批评分
bash score.sh --run outputs/smoke
```

评分与Qwen相同，报告同序30项指标及HTML、Markdown、LaTeX三种TEDS。
METRICS.md及SUMMARY.tsv位于各运行目录metrics下。评分失败和缺失项不冒充0分，ToTTo不伪造本地分数。
README只说明预期运行方式；原始模型真实四卡加载尚未在机器2验证。


## 2026-11-26 评分更新

默认 `bash score.sh` 使用策略进化V2规范化口径，结果写入 `metrics_normalized/`。
原始口径使用 `bash score.sh --normalization raw`，仍写入 `metrics/`。
规则、输出和小批评分命令见 [NORMALIZATION.md](NORMALIZATION.md)。
