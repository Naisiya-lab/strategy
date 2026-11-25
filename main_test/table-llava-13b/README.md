# Table-LLaVA-13B 原始版

复用已跑通的7B环境 `$HOME/.conda/envs/mmtab-table-llava`，不需要重新安装推理依赖。
默认复用 `../table-llava-7b/vendor/Table-LLaVA` 官方源码；如路径不同，修改config.json中的official_repo。
只支持作者原始版LlavaLlamaForCausalLM，不是HF转换版。配置中的模型路径：
`/nc1test1/lwj/Models/table-llava-v1.5-13b`。
CLIP共用 `/nc1test1/lwj/Models/clip-vit-large-patch14-336`。

```bash
cd /nc1test1/lwj/strategy/main_test/table-llava-13b
bash test.sh check
# 四张卡先释放，再启动；无需HTTP服务
bash test.sh smoke
# 小批确认后
bash test.sh full
```

四张A800各一个完整13B副本，FP16、贪心、llava_v1、1024输出上限，与7B相同。
原生上下文不足时明确记录实际预算缩减，输入无法容纳时记失败，不擅自截断原题。
同seed同组抽样368条；tqdm分卡显示；逐题保存原始回答、生成token IDs、用量、耗时、截断与异常。
outputs/smoke或outputs/full独立存储，不读取/覆盖7B结果。重复运行支持断点续跑。
配置/代码改变后另起--run-name，不混写旧结果。修改后的路径只写prepared_model配置副本，不改变原始权重。

```bash
# 若评分环境已经配置可跳过install_score
bash install_score.sh
bash score.sh --run outputs/smoke
bash score.sh
```

评分口径和7B一致，METRICS.md/SUMMARY.tsv按指定30项顺序输出，HTML/Markdown/LaTeX分别TEDS。
真实四卡13B显存和推理尚待机器2验证；本地只做数据/配置测试。
