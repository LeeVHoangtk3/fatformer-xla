"""Bind benchmark, real-image verification and latency to the same package."""
import argparse
import json
import math
from pathlib import Path

from tools.calibration_data import sha256_file
from tools.phase7_corrected import write_json


def finalize(benchmark, verification, latency, output):
    paths=[Path(p) for p in (benchmark,verification,latency)]
    reports=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    b,v,l=reports
    if l.get('schema')!='unified_latency/v3':
        raise ValueError('Require the current allocated-GPU latency protocol')
    package=b.get('package_sha256')
    if (b.get('schema')!='phase7_corrected_benchmark/v1' or b.get('status')!='completed'
            or not package or v.get('checkpoint_sha256')!=package or l.get('checkpoint_sha256')!=package):
        raise ValueError('All completed reports must refer to the same actual package')
    if v.get('real_image_acceptance') is not True or v.get('tensors_equal') is not True:
        raise ValueError('Missing real-image and tensor verification')
    for name,expected in b['artifacts'].items():
        if sha256_file(paths[0].parent/name)!=expected:
            raise ValueError(f'Benchmark artifact changed: {name}')
    is_cuda=str(l.get('device','')).startswith('cuda') and isinstance(l.get('gpu_name'),str)
    is_t4=is_cuda and 'T4' in l['gpu_name']
    observed=l.get('batches',{}).get('1',{}).get('overhead_delta_ms')
    latency_status='pending_gpu_measurement'
    if is_cuda and l.get('batches',{}).get('1',{}).get('status')=='out_of_memory':
        latency_status='insufficient_memory'
    elif is_cuda:
        if not isinstance(observed,(int,float)) or not math.isfinite(observed):
            raise ValueError('GPU report missing finite batch-1 overhead')
        latency_status='pass' if observed<5 else 'fail'
    result=dict(schema='phase7_corrected_acceptance/v1',package_sha256=package,
                report_integrity='verified',scientific_status='post_hoc_confirmation',
                G72=b['gates']['G7.2'],
                G73=dict(status='fail' if latency_status in ['fail','insufficient_memory'] else 'pending',
                         one_checkpoint_forward=True,real_image_equivalence=True,
                         hardware_protocol='any_compatible_cuda_gpu_with_sufficient_memory',
                         allocated_gpu_overhead=latency_status,
                         original_t4_reference=latency_status if is_t4 else 'not_measured_on_T4',
                         observed_gpu=l.get('gpu_name'),
                         observed_overhead_ms=observed,
                         condition_best_comparison='pending_verified_reference',
                         note='No claim of G7.3 PASS without the best-by-condition comparison.'),
                source_reports={str(p):sha256_file(p) for p in paths})
    write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['benchmark','verification','latency','output']:
        parser.add_argument('--'+key,required=True)
    print(json.dumps(finalize(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
