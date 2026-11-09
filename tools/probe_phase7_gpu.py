"""Find a batch that actually fits the allocated CUDA GPU before the long run."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.canonical_loader import load_canonical
from tools.calibration_data import sha256_file
from tools.phase7_corrected import TRAINING_SHA,write_json


def select_fitting_batch(attempt,candidates):
    records=[]
    for batch in sorted(set(candidates),reverse=True):
        if batch<1:
            raise ValueError('Batch sizes must be positive')
        result=None
        try:
            result=attempt(batch)
        except torch.OutOfMemoryError:
            records.append(dict(batch_size=batch,status='out_of_memory'))
        if result is not None:
            records.append(dict(batch_size=batch,status='fits',**result))
            return batch,records
        # Release the failed frame before clearing the allocator cache.
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    raise RuntimeError('No batch fits, including batch 1; free VRAM or use a GPU with more memory')


def probe(checkpoint,output,max_batch_size=16):
    if max_batch_size<1:raise ValueError('max_batch_size must be positive')
    if not torch.cuda.is_available():raise RuntimeError('A compatible CUDA GPU is required')
    if sha256_file(checkpoint)!=TRAINING_SHA:raise ValueError('Wrong training checkpoint')
    device=torch.device('cuda')
    free,total=torch.cuda.mem_get_info(device)
    report=dict(schema='phase7_gpu_memory_probe/v1',status='running',gpu_name=torch.cuda.get_device_name(device),
                cuda_capability=list(torch.cuda.get_device_capability(device)),
                total_vram_gib=total/(1024**3),free_vram_gib_before_model=free/(1024**3),
                precision='fp16_autocast',requested_max_batch=max_batch_size,
                purpose='Synthetic input tests memory only; no accuracy or latency acceptance')
    write_json(output,report)
    args=SimpleNamespace(backbone='CLIP:ViT-L/14',clip_path=None,num_classes=2,
                         use_srm=True,use_gating=True,num_vit_adapter=3,num_context_embedding=8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_=load_canonical(checkpoint,args)
    model=model.to(device).eval()
    def attempt(batch):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            x=torch.zeros(batch,3,224,224,device=device)
            _=model(x)
            torch.cuda.synchronize(device)
        return dict(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1024**3))
    candidates=[]
    batch=max_batch_size
    while batch:
        candidates.append(batch);batch//=2
    selected,records=select_fitting_batch(attempt,candidates)
    report.update(status='completed',selected_batch_size=selected,attempts=records)
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max_batch_size',type=int,default=16)
    args=parser.parse_args()
    try:
        print(probe(**vars(args)))
    except Exception as error:
        path=Path(args.output)
        report=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        report.update(status='failed',error=str(error),synthetic_memory_probe_only=True)
        write_json(path,report)
        raise


if __name__=='__main__':main()
