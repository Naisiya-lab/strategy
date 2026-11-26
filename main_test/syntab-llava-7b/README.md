# SynTab-LLaVA-7B 双视觉编码器基线

原图与原题直接作答，不使用本课题策略、识表、子表、核查或返工。
模型目录默认 `/nc1test1/lwj/Models/SynTab-LLaVA-v1.5-7B-Conv-Clip`。
四卡各一个完整模型副本，四条tqdm、固定seed同批368条小样本、逐题保存与断点续跑。

## 安装与运行

SynTab与Table-LLaVA虽然都导入llava，但代码和依赖不同，必须分环境。
独立环境 `$HOME/.conda/envs/mmtab-syntab`，Python3.10、torch2.1.2、transformers4.37.2。
源码克隆到本目录vendor/SynTab-LLaVA，commit、环境依赖记录到logs。首次安装需联网。
使用官方代码的双processor、highres_images/lowres_images与image_sizes输入，不套用普通LLaVA单图预处理。

```bash
cd /nc1test1/lwj/strategy/main_test/syntab-llava-7b
bash install.sh
bash test.sh check
# 确认四张卡没有其他推理任务占用后执行
bash test.sh smoke
# 小批检查通过后
bash test.sh full
```

不启动HTTP服务，不需要start.sh/health。安装不会更改Qwen、Table-LLaVA或评分环境。
不安装训练extras/deepspeed。FlashAttention未强制启用；若官方源码导入提示缺少flash_attn，且机器具备CUDA编译工具链，可执行：

```bash
INSTALL_FLASH_ATTN=1 bash install.sh
```

## 两个本地视觉编码器

官方checkpoint目录结构包含：

```text
SynTab-LLaVA-v1.5-7B-Conv-Clip/
  config.json
  model-00001-of-00004.safetensors 等
  ConvLLaVA-ConvNeXt-1536/
  clip-vit-large-patch14-336/
```

机器2实际CLIP目录为`clip-vit-large-patch14-336`。官方README示例曾写`Clip-vit-largr-patch14-336`，不能照搬为实际路径；Linux区分大小写。config.json的highres_vision_path、lowres_vision_path应填写本机真实目录。
check要求两个视觉目录包含config.json，低分辨率CLIP还需preprocessor_config.json。
使用prepared_model中的符号链接和独立配置副本重定向双视觉路径，不改共享原始权重。
推理强制离线，缺权重不会后台下载。只支持完整模型，检测到adapter_config.json会拒绝启动并说明需要完整checkpoint。

## 生成和上下文

默认llava_v1、FP16、贪心、max_new_tokens=1024，与官方eval/model_vqa.py相同；不将Qwen的8192写成此模型配置。
融合后的图文长度从官方准备函数实际返回的embeddings获取，不用普通CLIP的576-token估计。
若输入加1024输出预算超出原生max_position_embeddings，记显式错误，不静默裁剪原题或缩短输出。
不同于Table-LLaVA已有脚本的预算缩减规则，SynTab采用固定1024；结果须按实际配置注明。
如果源码接口变化、不能识别融合长度，拒绝该题而不伪造token用量，先检查小批。
保留generation token IDs、模型回答、finish_reason、usage、seconds和错误。
判定length不重新采样。支持同命令断点续跑，以及`--retry-errors`、`--run-name`。
官方加载失败为整轮失败，已保存的逐题结果保留。不同输入/源码/配置会拒绝混入同一实验目录。

## 评分和结果

```bash
bash install_score.sh
bash score.sh --run outputs/smoke
bash score.sh
# 已有完整metrics.json时只重排
bash score.sh --report-only
```

结果位于outputs/smoke或outputs/full，评分在对应metrics目录。
沿用Qwen/Table-LLaVA评分公式和固定30项顺序，包含HTML/Markdown/LaTeX TEDS，METRICS.md与SUMMARY.tsv。
缺失题拒绝评分；失败记录空答案、不删除；不可评分项为“—”；RCE沿用官方计数局限；ToTTo测试无本地参考时不伪造BLEU。
环境安装和真实四卡GPU推理尚未在机器2验证。本地测试只验证文件路径、配置隔离、融合长度识别、数据及评分逻辑。

来源：https://github.com/bang123-box/SynTab-LLaVA
