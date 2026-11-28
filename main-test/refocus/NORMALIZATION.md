# MMTab Strategy V2 规范化评分

2026-11-26：默认评分入口已对齐“策略进化V2”的主评测规则，包括最新 scalar-unit v2 修正。
这是规范化评分协议，不能把结果标记成未经处理的官方原始分。

```bash
# 重算完整推理结果（不启动模型）
bash score.sh
# 小批结果：替换为真实运行目录
bash score.sh --run outputs/smoke
# 保留原始官方评分口径
bash score.sh --normalization raw
```

默认输出在运行目录的 `metrics_normalized/`：
- `METRICS.md`：按指定论文表格顺序排列。
- `SUMMARY.tsv`、`TQA_TFV_T2T.tsv`、`TSU_TR.tsv`：可粘贴进Excel。
- `metrics.json`：分组指标、分母、执行失败、截断、来源及规范化版本。
- `normalization_detail.json`：逐题原始/规范化文本、参考处理和QA规则记录。
- `teds_per_sample.json`：TR逐题分数及解析失败。

`--normalization raw`仍写入原来的 `metrics/`；默认重算不覆盖旧原始分。原始推理文件始终不变。
安装依赖沿用原来的 `install_score.sh` / mmtab-score 环境，无需重新安装模型环境。
`--report-only`默认重排规范化结果；重排原始结果使用 `--normalization raw --report-only`。

## 规则边界

预测和参考对称应用相同规则，规范化模块不导入实验运行器、不调用模型：
- QA：明确answer字段、无冲突的答案包装、选项文本、Unicode/大小写/空白及数字书写形式。
- 单位：仅按题目与表头的明确依据处理；重复数值说明必须与原数值相等。
- TFV：明确肯定/否定/未知标签及允许的选项包装。
- 结构：字段顺序、坐标列表/元组表达、文本值清理及明确的空合并格列表。
- T2T/FeTaQA：按V2主评测剥离明确包装，BLEU继续保留大小写；忽略大小写是另一个诊断口径，本入口不混入主分。
- TR：原始预测不规范化，HTML / Markdown / LaTeX分别计算原有TEDS。

不修正数值、坐标、漏格或参考标注，不推断答案，不转换数量级，不把变化量当变化率。
保持官方RCE等任务的原有公式与解析局限，所有失败样本保留分母；ToTTo无公开测试参考时不本地计分。
`normalization.py`记录所有提取来源的内容哈希，五个模型使用同一份快照；可独立复制单个模型目录运行。
