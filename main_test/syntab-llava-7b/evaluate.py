"""Four single-GPU workers using SynTab's dual-encoder loader and prompt."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from data import ROOT, digest, prepare, save


def model_view(cfg):
    source=Path(cfg['model_path']).resolve()
    if (source/'adapter_config.json').exists():
        raise ValueError('LoRA adapter detected: this runner requires the full SynTab checkpoint')
    original=json.loads((source/'config.json').read_text())
    if original.get('architectures')!=['LlavaLlamaForCausalLM']:
        raise ValueError('Expected SynTab LlavaLlamaForCausalLM full checkpoint')
    overrides={}
    for field,config_key in [('mm_highres_vision_tower','highres_vision_path'),
                             ('mm_lowres_vision_tower','lowres_vision_path')]:
        path=Path(cfg[config_key]).resolve()
        if not (path/'config.json').is_file():
            raise ValueError('Vision encoder config missing: '+str(path))
        overrides[field]=str(path)
    if not (Path(cfg['lowres_vision_path'])/'preprocessor_config.json').is_file():
        raise ValueError('Low-resolution CLIP processor config missing')
    key=digest(json.dumps([str(source),original,overrides],sort_keys=True).encode())[:16]
    view=ROOT/'prepared_model'/key/'SynTab-LLaVA-v1.5-7B-Conv-Clip';view.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.name not in ('config.json','.cache'):
            target=view/p.name
            if not target.exists(): target.symlink_to(p,target_is_directory=p.is_dir())
    original.update(overrides)
    save(view/'config.json',original)
    return str(view)


def expanded_prompt_length(prepared):
    # Common LLaVA contract: tuple contains rank-3 inputs_embeds. Read actual
    # fused sequence length rather than assuming CLIP's 576 tokens for SynTab.
    tensors=[v for v in prepared if getattr(v,'ndim',None)==3]
    if len(tensors)!=1:
        raise ValueError('Cannot identify fused image/text embeddings from SynTab preparation')
    return int(tensors[0].shape[1])


def sample_path(out,sid):return Path(out)/'samples'/(digest(sid.encode())+'.json')


def worker(job):
    index,gpu,rows,cfg,view,out=job
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from PIL import Image
    from tqdm import tqdm
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    torch.manual_seed(cfg['seed']);disable_torch_init()
    tokenizer,model,processor,context_len=load_pretrained_model(view,None,'SynTab-LLaVA-v1.5-7B-Conv-Clip')
    if not isinstance(processor,(list,tuple)) or len(processor)!=2:
        raise ValueError('Expected high-resolution and low-resolution processors')
    observed={}
    original_prepare=model.prepare_inputs_labels_for_multimodal
    def capture_preparation(*args,**kwargs):
        result=original_prepare(*args,**kwargs)
        length=expanded_prompt_length(result)
        observed['prompt_tokens']=length
        limit=getattr(model.config,'max_position_embeddings',context_len)
        observed['context_limit']=limit
        if length+cfg['max_tokens']>limit:
            raise ValueError(f'Fused prompt {length} + output budget {cfg["max_tokens"]} exceeds native context {limit}; no silent truncation')
        return result
    model.prepare_inputs_labels_for_multimodal=capture_preparation
    for row in tqdm(rows,desc=f'GPU {gpu}',position=index,unit='sample',dynamic_ncols=True):
        started=time.monotonic();record=dict(row,status='error',gpu=str(gpu))
        observed.clear()
        try:
            data=Path(row['image']).read_bytes()
            if digest(data)!=row['image_sha256']:raise ValueError('Image changed after preflight')
            marker=DEFAULT_IMAGE_TOKEN
            if getattr(model.config,'mm_use_im_start_end',False):
                marker=DEFAULT_IM_START_TOKEN+marker+DEFAULT_IM_END_TOKEN
            conv=conv_templates[cfg['conv_mode']].copy()
            conv.append_message(conv.roles[0],marker+'\n'+row['question']);conv.append_message(conv.roles[1],None)
            prompt=conv.get_prompt()
            ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
            with Image.open(row['image']) as image:
                image=image.convert('RGB');image_size=image.size
                high=process_images([image],processor[0],model.config)[0]
                low=process_images([image],processor[1],model.config)[0]
            observed.clear()
            budget=cfg['max_tokens']
            record.update(text_input_tokens=int(ids.shape[1]),requested_max_tokens=budget,effective_max_tokens=budget)
            with torch.inference_mode():
                output=model.generate(ids,
                    highres_images=high.unsqueeze(0).half().cuda(),
                    lowres_images=low.unsqueeze(0).half().cuda(),
                    image_sizes=[image_size],do_sample=False,num_beams=1,
                    max_new_tokens=budget,use_cache=True)
            if 'prompt_tokens' not in observed:
                raise ValueError('SynTab did not expose fused prompt length; cannot verify context')
            prompt_tokens=observed['prompt_tokens']
            record.update(observed)
            if output.shape[1]>=ids.shape[1] and torch.equal(output[:,:ids.shape[1]],ids):
                generated=output[:,ids.shape[1]:]
            else: generated=output
            text=tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip()
            stop=conv.sep if conv.sep_style!=SeparatorStyle.TWO else conv.sep2
            if text.endswith(stop):text=text[:-len(stop)].strip()
            count=generated.shape[1];last=int(generated[0,-1]) if count else None
            truncated=count>=budget and last!=tokenizer.eos_token_id
            record.update(answer=text,generated_token_ids=generated[0].tolist(),
                          finish_reason='length' if truncated else 'stop',
                          usage={'prompt_tokens':prompt_tokens,'completion_tokens':count,'total_tokens':prompt_tokens+count})
            if not text:raise ValueError('Empty answer preserved; not regenerated')
            record['status']='ok'
        except Exception as exc:
            record.update(observed)
            record['error']=type(exc).__name__+': '+str(exc)
            torch.cuda.empty_cache()
        record['seconds']=time.monotonic()-started
        save(sample_path(out,row['id']),record)
    return len(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['check','smoke','full'])
    ap.add_argument('--run-name');ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--per-group',type=int,default=4)
    args=ap.parse_args();cfg=json.loads((ROOT/'config.json').read_text())
    if args.per_group<1:ap.error('per-group must be positive')
    if len(set(cfg['gpu_ids']))!=len(cfg['gpu_ids']):raise ValueError('Duplicate GPU IDs')
    rows,ann_hash=prepare(cfg,args.mode,args.per_group)
    view=model_view(cfg);print('Preflight OK:',len(rows),'samples; dual vision encoders configured',flush=True)
    if args.mode=='check':return
    name=args.run_name or args.mode
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Invalid run name')
    out=ROOT/'outputs'/name;out.mkdir(parents=True,exist_ok=True)
    lock=out/'.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        official=ROOT/'vendor/SynTab-LLaVA'
        commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
        model_config=Path(cfg['model_path'],'config.json').read_bytes()
        manifest={'config':cfg,'rows':rows,'mode':args.mode,'annotation_sha256':ann_hash,
                  'official_commit':commit,'model_config_sha256':digest(model_config),
                  'script_sha256':digest(Path(__file__).read_bytes()),'data_script_sha256':digest((ROOT/'data.py').read_bytes())}
        mf=out/'manifest.json'
        if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Inputs changed; use a new run-name')
        save(mf,manifest);pending=[]
        for r in rows:
            p=sample_path(out,r['id']);old=json.loads(p.read_text()) if p.exists() else None
            if old is None or (args.retry_errors and old['status']!='ok'):
                if old:save(out/'previous_attempts'/(p.stem+'_'+str(time.time_ns())+'.json'),old)
                pending.append(r)
        started=time.monotonic()
        if pending:
            with ProcessPoolExecutor(max_workers=len(cfg['gpu_ids']),mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs=[(i,gpu,pending[i::len(cfg['gpu_ids'])],cfg,view,str(out)) for i,gpu in enumerate(cfg['gpu_ids'])]
                futures=[pool.submit(worker,j) for j in jobs if j[2]]
                for future in as_completed(futures):future.result()
        records=[json.loads(sample_path(out,r['id']).read_text()) for r in rows]
        with (out/'predictions.jsonl').open('w',encoding='utf8') as f:
            for r in records:f.write(json.dumps({'question_id':r['id'],'text':r.get('answer','') if r['status']=='ok' else ''},ensure_ascii=False)+'\n')
        summary={'total':len(rows),'ok':sum(r['status']=='ok' for r in records),'errors':sum(r['status']!='ok' for r in records),
                 'truncated':sum(r.get('finish_reason')=='length' for r in records),'newly_attempted':len(pending),
                 'session_seconds':time.monotonic()-started,'context_limited':sum(r.get('effective_max_tokens',cfg['max_tokens'])<cfg['max_tokens'] for r in records)}
        save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    finally:lock.unlink()


if __name__=='__main__':main()
