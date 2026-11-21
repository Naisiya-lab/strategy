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
若出现driver/PTX错误，保留日志并适配驱动或安装方案；仅pip安装成功不代表服务可用。
安装和真实GPU推理尚未在机器2验证。

## 默认路径

- 权重：`/nc1test1/lwj/Models/Qwen3-VL-8B-Instruct`
- 标注：`/nc1test1/lwj/data/MMTab_eval/MMTab-eval_test_data_49K.json`
- 图片：递归扫描 `/nc1test1/lwj/data/MMTab_eval/images`，自动兼容 `images/all_test_image` 等层级。
- 配置：本目录 `config.json`；端口与start.sh对应，改端口时需同步两处。
- 环境：`$HOME/.conda/envs/mmtab-qwen`；自定义环境时使用 `PYTHON=/path/to/python bash start.sh` 或 `bash test.sh ...`。

当前启动参数：BF16、每卡一副本、上下文16384、输出上限8192。图片处理沿用权重中的processor配置，不人为降低分辨率。
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
运行汇总不是准确率。`predictions.jsonl`用于后续公共评分器或官方notebook分项评分；本目录未另写一套不一致的准确率算法。
小批选题参考任务分组只用于抽样，不作为模型输入。

## 停止和本地验证

```bash
# 只停止本目录PID与端口匹配的vLLM服务
bash stop.sh

# 无需模型/GPU，模拟HTTP接口验证并发请求、原题输入、断点续跑、截断与配置冲突
python -m unittest discover -s . -p test_evaluate.py -v
```

首次编译/模型加载不计入推理计时；首轮推理仍可能存在预热成本。
吞吐数字包括客户端图像读取和请求时间；单题seconds包含传输重试，不能等同GPU纯计算时间。
