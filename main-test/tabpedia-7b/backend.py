"""TabPedia official model and image preprocessing; no gold answers enter prompts."""
import ast
import importlib.util
import os
from pathlib import Path
import sys


class Backend:
    def __init__(self, cfg):
        import torch
        from mmengine.config import Config
        from xtuner.registry import BUILDER
        self.torch = torch
        self.cfg = cfg
        code = Path(cfg['code_path']).resolve()
        sys.path.insert(0, str(code))
        os.chdir(code)
        spec = importlib.util.spec_from_file_location('tabpedia_official_eval', code / 'eval/eval_TabPedia.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        # Remove unused training-only imports from the released inference config.
        source = (code / 'tools/configs/Internlm2_7b_chat_TabPedia.py').read_text()
        tree = ast.parse(source)
        excluded = set()
        names = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('xtuner'):
                excluded.update(range(node.lineno, node.end_lineno + 1))
                names.update(a.asname or a.name for a in node.names)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if names & used:
            raise ValueError('Official config now uses training imports; adaptation must be reviewed')
        source = ''.join(line for i, line in enumerate(source.splitlines(True), 1) if i not in excluded)
        config = Config.fromstring(source, '.py')
        config.tokenizer.pretrained_model_name_or_path = cfg['llm_path']
        config.model.llm.pretrained_model_name_or_path = cfg['llm_path']
        config.model.high_res_visual_encoder.pretrained_model_name_or_path = cfg['donut_path']
        config.model.low_res_visual_encoder.pretrained_model_name_or_path = cfg['clip_path']
        config.global_image_processor.pretrained_model_name_or_path = cfg['clip_path']
        config.model.tokenizer = config.tokenizer
        self.tokenizer = BUILDER.build(config.tokenizer)
        self.tokenizer.add_tokens(config.special_tokens, special_tokens=False)
        self.model = official.TabPediaModel(**config.model)
        state = torch.load(cfg['model_path'], map_location='cpu')
        # Refuse a partially loaded baseline rather than silently scoring random weights.
        result = self.model.model.load_state_dict(state, strict=False)
        print('missing keys:', result.missing_keys, flush=True)
        print('unexpected keys:', result.unexpected_keys, flush=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError('Checkpoint mismatch: review keys before inference')
        del state
        self.model.eval().cuda()
        self.processor = object.__new__(official.TabPedia_dataset)
        self.processor.high_res_size = [config.high_res_image_height, config.high_res_image_width]
        self.processor.image_transform = None
        self.processor.global_image_processor = BUILDER.build(config.global_image_processor)
        self.processor.to_tensor = official.T.Compose([
            official.T.ToTensor(), official.T.Normalize(official.IMAGENET_DEFAULT_MEAN, official.IMAGENET_DEFAULT_STD)])
        self.template = config.prompt_template
        self.context = min(config.max_length, self.model.model.llm.config.max_position_embeddings)

    def predict(self, row):
        from PIL import Image
        torch = self.torch
        marker = '<im_start><C_IMG>' + '<im_patch>' * 256 + '<T_IMG>' + '<im_patch>' * 1200 + '<im_end>'
        question = row['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        question = question.replace('<image>', marker)
        prompt = self.template.SYSTEM + self.template.INSTRUCTION.format(input=question)
        prompt += ''.join(f'<extra_token_{i}>' for i in range(256))
        ids = self.tokenizer(prompt, return_tensors='pt', truncation=False).input_ids.cuda()
        count = ids.shape[1]
        budget = min(self.cfg['max_tokens'], self.context - count)
        if budget < 1:
            raise ValueError(f'Input tokens {count} exceed context {self.context}; no input truncation')
        with Image.open(row['image']) as im:
            high, low = self.processor.process_image(im.convert('RGB'))
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                high_res_image=high.unsqueeze(0).to(device='cuda', dtype=torch.bfloat16),
                low_res_image=low.unsqueeze(0).to(device='cuda', dtype=torch.bfloat16),
                max_new_tokens=budget, do_sample=False, num_beams=1, use_cache=True,
                repetition_penalty=self.cfg['repetition_penalty'],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        generated = output[0, count:].tolist()
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer, dict(generated_token_ids=generated, prompt_tokens=count,
            requested_max_tokens=self.cfg['max_tokens'], effective_max_tokens=budget,
            finish_reason='length' if len(generated) >= budget and
            (not generated or generated[-1] != self.tokenizer.eos_token_id) else 'stop')
