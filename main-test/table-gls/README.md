# Table-GLS MMTab runner

两个配置：qwen3-vl-8b（GPU 0，TP=1）、qwen3-vl-32b（GPU 0,1，TP=2）。逐样本顺序推理，优先保证可恢复性；不是吞吐量优化实现。不要同时运行两种模型。

官方源码快照见 official/PROVENANCE.json。使用官方三阶段提示词、Qwen原生模板及输入分支；阶段输出预算1024/1024/4096。max_model_len=32768（官方脚本为86256），显存比例0.8，seed=42；属于明确记录的运行配置差异。全MMTab任务套用官方task分支属于扩展评测范围，不能称为官方原论文全任务复现。

## 运行
在容器激活 tablegls，进入本目录：

```bash
conda activate tablegls
cd /nc1test1/lwj/strategy/main-test/table-gls
bash qwen3-vl-8b/test.sh check
bash qwen3-vl-8b/test.sh smoke
bash qwen3-vl-8b/test.sh full
```

32B替换目录名即可。check只检查数据，不加载模型。数据默认 /nc1test1/lwj/data/MMTab_eval，图片放在其images目录（递归检索）；配置可修改。smoke默认每任务/来源组合1条，--per-group可改。更改实验参数后使用 --run-name 新名称。

## 保存和恢复
每个样本的每个阶段都原子写入并fsync：outputs/<run>/stages/{1,2,3}/<id哈希>.json。最终结果写入samples/；完成后导出predictions.jsonl和summary.json。原始输出、提示词、终止原因、token数量保留。显式且无冲突的answer字段提取为评分答案；未提取时保留原文，不猜测答案。

重复同一命令跳过已完成样本，未完成样本复用已有阶段。失败后立即停止，错误写入samples；用原命令加 --retry-errors 重试错误样本。错误记录先归档。Ctrl+C/进程被杀时，最多重做尚未落盘的当前阶段。Linux flock在崩溃后自动释放，无需删除.lock文件。

恢复严格核对配置、标注、图片哈希、样本清单和运行源码。模型权重按路径引用，运行期间不得替换同路径权重。stage length会记录但沿用官方行为继续后续阶段；评分报告最终阶段截断情况，阶段详情可逐条检查。

## 评分
评分代码及规范化直接复制现有Qwen主实验实现；TR按HTML/Markdown/LaTeX分别计算TEDS。结构化answer字段在推理导出时解包，以便TR评分读取真正表格；raw模式指不进行V2规范化，仍基于已解包的最终答案。

```bash
bash install_score.sh
```

注意score.sh会切换到模型目录，建议使用绝对路径指定--run：

```bash
bash qwen3-vl-8b/score.sh --run /nc1test1/lwj/strategy/main-test/table-gls/qwen3-vl-8b/outputs/smoke
bash qwen3-vl-8b/score.sh
```

默认完整结果outputs/full，规范化结果metrics_normalized/，原始规则--normalization raw写metrics/。同一模式重复评分更新该模式指标；预测不改。TEDS缓存支持重复运行复用。未完成推理不允许评分；已记录失败按空答案计入，不丢样本。ToTTo缺少公开测试参考时沿用现有脚本标记不可本地评分。

## 验证
python -m unittest test_resume test_score -v
本地覆盖提示词与官方逐字符一致、阶段中断恢复、已完成阶段不重复调用及评分回归。实际A800加载和推理仍需在容器验证。
