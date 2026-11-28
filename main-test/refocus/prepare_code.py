"""CPU-only download of pinned official Phi custom code; no weights downloaded."""
import json
from pathlib import Path
from huggingface_hub import HfApi
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from evaluate import save
root = Path(__file__).resolve().parent
cfg = json.loads((root/'config.json').read_text())
repo = 'microsoft/Phi-3.5-vision-instruct'
revision = cfg.get('code_revision') or HfApi().model_info(repo).sha
for ref in ('configuration_phi3_v.Phi3VConfig', 'modeling_phi3_v.Phi3VForCausalLM',
            'processing_phi3_v.Phi3VProcessor'):
    get_class_from_dynamic_module(ref, repo, revision=revision)
cfg['code_revision'] = revision
save(root/'config.json', cfg)
print('Phi custom code ready:', revision)
