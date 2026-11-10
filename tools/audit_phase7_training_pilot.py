"""Independently audit/import submitted pilot logs without loading missing weights."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import fnmatch
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tarfile

import yaml

from tools.phase7_git_source import SOURCE_PATHS

ROOT=Path(__file__).resolve().parents[1]
ARMS=('p0_legacy_full','p1_clip_full')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-9)


class Checks:
    def __init__(self):self.groups={};self.failures=[]
    def need(self,name,condition,detail=''):
        group=self.groups.setdefault(name,dict(checked=0,failed=0))
        group['checked']+=1
        if not condition:
            group['failed']+=1
            if len(self.failures)<30:self.failures.append(dict(check=name,detail=str(detail)))
    def finish(self):
        if self.failures:raise ValueError('Pilot evidence audit failed: '+json.dumps(self.failures,ensure_ascii=False))


def inspect_trace(path,check):
    events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    previous=None
    for event in events:
        check.need('trace_schema',event['schema']=='optimizer-update/v1')
        check.need('trace_no_unrecoverable',event['unrecoverable'] is False)
        check.need('trace_no_applied_nonfinite',not event['applied'] or event['grad_norm_finite'])
        check.need('trace_finite_norm_contract',
            (event['grad_norm'] is not None and math.isfinite(event['grad_norm'])) if event['grad_norm_finite'] else event['grad_norm'] is None)
        before,after=event['amp_scale_before'],event['amp_scale_after']
        check.need('trace_positive_amp_scale',all(v is not None and math.isfinite(v) and v>0 for v in [before,after]))
        if previous is not None:check.need('trace_amp_scale_continuity',before==previous)
        previous=after
        check.need('trace_amp_skip_matches_call',event['amp_skipped']==(not event['applied']))
        if event['amp_skipped']:check.need('trace_overflow_halves_scale',after==before/2)
    return events


def verify_submission(source,notebook,repo=ROOT):
    source,notebook,repo=Path(source).resolve(),Path(notebook).resolve(),Path(repo).resolve()
    check=Checks();selection=read(source/'selection.json');status=read(source/'status.json')
    provenance=read(source/'source_identity.json');commit=provenance['commit']
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Invalid source commit')
    # git archive applies working-tree EOL rules. Disable Windows CRLF conversion
    # to compare canonical Git bytes with the Linux Colab checkout, not edited files.
    tree=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines()
    names=[name for name in tree if any(name==spec or name.startswith(spec+'/') or fnmatch.fnmatchcase(name,spec) for spec in SOURCE_PATHS)]
    data=subprocess.check_output(['git','-c','core.autocrlf=false','archive','--format=tar',commit,'--',*names],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        pinned={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in pinned.items()}
    check.need('git_file_set',set(hashes)==set(provenance['files']))
    for name,value in provenance['files'].items():check.need('git_file_hash',hashes.get(name)==value,name)
    fingerprint=hashlib.sha256(json.dumps(provenance['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check.need('git_fingerprint',fingerprint==provenance['tracked_files_fingerprint'])
    src=hashlib.sha256()
    for name in sorted(name for name in pinned if name.startswith('src/') and name.endswith('.py')):
        src.update(name.encode());src.update(pinned[name])
    code_hash=src.hexdigest()
    n=read(notebook);cells=[c for c in n['cells'] if c['cell_type']=='code']
    check.need('notebook_four_code_cells',len(cells)==4)
    check.need('notebook_execution_order',[c.get('execution_count') for c in cells]==[1,2,3,4])
    for c in cells:
        check.need('notebook_no_error_outputs',not any(o.get('output_type')=='error' for o in c.get('outputs',[])))
        check.need('notebook_execution_status',c.get('metadata',{}).get('executionInfo',{}).get('status')=='ok')
    notebook_output='\n'.join(''.join(o.get('text',[])) for c in cells for o in c.get('outputs',[]))
    check.need('notebook_contains_commit',commit in notebook_output)
    notebook_summary=dict(sha256=sha(notebook),code_cells=len(cells),execution_counts=[c.get('execution_count') for c in cells],
        smoke_wall_minutes=cells[2]['metadata']['executionInfo']['elapsed']/60000,
        train_wall_minutes=cells[3]['metadata']['executionInfo']['elapsed']/60000,
        note='Cell wall times include preflight/smoke/validation/backups; epoch CSV time is a different scope.')
    manifest=read(repo/'DATASET/manifests/data_manifest_v2.json')
    label_counts=Counter(r['label'] for r in manifest['records'] if r['split']=='val_select')
    check.need('validation_class_counts',label_counts=={0:590,1:490})
    expected_inputs=json.loads(pinned['DATASET/logs/phase7/improvement_round1/preparation/expected_inputs.json'])
    summaries={};contexts={};configs={};references={};run_ids=set()
    for arm in ARMS:
        folder=source/arm;identity=read(folder/'identity.json');context=identity['context'];contexts[arm]=context
        cfg=yaml.safe_load((folder/'run_config.yaml').read_text(encoding='utf-8'));configs[arm]=cfg
        run_id=Path(cfg['training']['save_dir']).parent.name
        if not re.fullmatch(r'pilot_\d{8}_\d{6}',run_id):raise ValueError('Invalid run ID')
        run_ids.add(run_id);check.need('notebook_contains_run_id',run_id in notebook_output)
        proof=read(folder/'run_verification.json');preflight=read(folder/'preflight.json')
        history=rows(folder/'train_history.csv');groups=rows(folder/'val_groups.csv')
        check.need('five_training_epochs',[int(r['epoch']) for r in history]==list(range(1,6)),arm)
        check.need('six_validation_epochs',[int(r['epoch']) for r in groups]==list(range(6)),arm)
        check.need('proof_context',proof['context']==context and proof['source_commit']==commit)
        check.need('identity_context',identity['source_commit']==commit and identity['source_fingerprint']==fingerprint)
        check.need('source_code_hash',context['source_sha256']==code_hash==preflight['source_sha256'])
        check.need('runner_hash',identity['runner_sha256']==hashes['tools/phase7_training_pilot.py'])
        check.need('preflight_tool_hash',preflight['preflight_tool_sha256']==hashes['tools/preflight_phase7_train.py'])
        check.need('preflight_config_hash',preflight['config_sha256']==sha(folder/'run_config.yaml'))
        check.need('preflight_all_pass',preflight['ready_to_train'] is True and all(v['status']=='PASS' for v in preflight['checks'].values()))
        check.need('protocol_fixed',context['epochs']==5 and context['seed']==42 and context['batch_size']==16 and context['grad_accum_steps']==2 and context['validation_preprocessing']=='clip' and context['val_split']=='val_select' and not context['dry_run'])
        for key,file in [('calibration_file_sha256','DATASET/logs/phase7/calibration/baseline.json'),('degradation_config_sha256','configs/degradation_v2.yaml')]:
            check.need('bound_input_hash',context[key]==hashes[file])
        check.need('manifest_hash',context['train_manifest_sha256']==sha(repo/'DATASET/manifests/data_manifest_v2.json'))
        check.need('bound_baseline_hash',context['checkpoint_init_sha256']==context['teacher_checkpoint_sha256']==expected_inputs['expected_assets_sha256']['fatformer_4class_ckpt.pth'])
        check.need('bound_clip_hash',context['clip_path_sha256']==expected_inputs['expected_assets_sha256']['ViT-L-14.pt'])
        for file in ['train_history.csv','val_groups.csv']:check.need('remote_verification_csv_hash',sha(folder/file)==proof['files'][file],arm+'/'+file)
        trace=inspect_trace(folder/'optimizer_updates.jsonl',check)
        check.need('training_trace_length',len(trace)==1915,arm)
        check.need('training_trace_scope',all(e['phase']=='training' for e in trace))
        per_epoch=[]
        for row in history:
            epoch=int(row['epoch']);events=[e for e in trace if e['epoch']==epoch]
            expected_boundaries=list(range(2,766,2))+[765]
            check.need('training_update_boundaries',[e['mini_batch'] for e in events]==expected_boundaries)
            check.need('training_attempt_sequence',[e['attempt'] for e in events]==list(range(1,384)))
            check.need('training_global_steps',all(e['global_step']==(epoch-1)*765+e['mini_batch'] for e in events))
            finite=[e['grad_norm'] for e in events if e['grad_norm_finite']]
            applied=sum(e['applied'] for e in events);skipped=len(events)-applied
            recomputed=dict(optimizer_steps_expected=383,optimizer_steps_attempted=len(events),optimizer_steps_applied=applied,
                optimizer_steps_skipped=skipped,nonfinite_grad_events=len(events)-len(finite),grad_norm_finite_count=len(finite),train_steps=765)
            for key,value in recomputed.items():check.need('epoch_counter_matches_trace',int(row[key])==value,arm+'/'+str(epoch)+'/'+key)
            check.need('epoch_mean_gradient',bool(finite) and close(row['grad_norm'],statistics.mean(finite)))
            check.need('epoch_amp_endpoints',float(row['amp_scale_start'])==events[0]['amp_scale_before'] and float(row['amp_scale_end'])==events[-1]['amp_scale_after'])
            check.need('epoch_stability_policy',applied>0 and skipped/len(events)<=.05)
            check.need('epoch_loss_finite',all(math.isfinite(float(row[k])) for k in ['loss_total','loss_ce','loss_kd','loss_cons','loss_feat','loss_guard','grad_norm']))
            check.need('epoch_loss_decomposition',math.isclose(float(row['loss_total']),.3*(float(row['ce_clean'])+float(row['ce_deg']))+.7*float(row['kd'])+.2*float(row['cons'])+.3*float(row['feat'])+.5*float(row['guard']),rel_tol=1e-5,abs_tol=1e-5))
            for key in ['clean_r','clean_f','q30_r','q30_f']:
                check.need('train_validation_csv_agree',close(row[key],groups[epoch][key]))
            check.need('realized_steps',int(row['train_steps'])==765)
            check.need('clean_accuracy_definition',close(row['val_acc'],100*(590*float(row['clean_r'])+490*float(row['clean_f']))/1080))
            per_epoch.append(dict(epoch=epoch,**recomputed,grad_norm_mean=statistics.mean(finite),skip_fraction=skipped/len(events),
                amp_scale_start=events[0]['amp_scale_before'],amp_scale_end=events[-1]['amp_scale_after']))
        for row in groups:
            values=[float(row[k]) for k in ['clean_r','clean_f','q30_r','q30_f']]
            check.need('worst_group_recomputed',close(row['worst'],min(values)))
            check.need('validation_operating_point',close(row['threshold'],.9127) and close(row['temperature'],1.6402))
            check.need('validation_sample_counts',int(row['n_eval_clean'])==1080 and int(row['n_eval_q30'])==300)
            for key,denominator in [('clean_r',590),('clean_f',490),('q30_r',150),('q30_f',150)]:
                count=float(row[key])*denominator
                check.need('recalls_have_integer_counts',0<=count<=denominator and close(count,round(count)))
        eligible=[r for r in groups if float(r['clean_r'])>=.9]
        selected=max(eligible,key=lambda r:float(r['worst']))
        chosen=selection['arms'][arm]
        check.need('selected_epoch_recomputed',int(selected['epoch'])==proof['selected_epoch']==chosen['epoch'])
        for key in ['clean_r','clean_f','q30_r','q30_f']:
            check.need('selection_matches_validation',close(chosen[key],selected[key]))
        check.need('selection_worst_matches_validation',close(chosen['worst_group'],selected['worst']))
        check.need('selected_model_hash_reference',chosen['sha256']==proof['files']['best.pth'] and bool(re.fullmatch('[0-9a-f]{64}',chosen['sha256'])))
        smoke=read(folder/'stability_smoke.json');decision=read(folder/'stability_decision.json')
        check.need('latest_smoke_hash',decision['smoke_sha256']==sha(folder/'stability_smoke.json'))
        check.need('smoke_context',smoke['context']==context and smoke['evidence_level']=='measured')
        smoke_traces=[];latest=None
        for path in sorted(folder.glob('smoke_*/optimizer_updates.jsonl')):
            events=inspect_trace(path,check);counts=Counter(e['phase'] for e in events)
            check.need('smoke_trace_horizon',counts=={'warmup':3,'benchmark':50})
            if sha(path)==decision['trace_sha256']:latest=[e for e in events if e['phase']=='benchmark']
            smoke_traces.append(dict(file=path.relative_to(folder).as_posix(),warmup_skips=sum(not e['applied'] for e in events if e['phase']=='warmup'),measurement_skips=sum(not e['applied'] for e in events if e['phase']=='benchmark')))
        check.need('latest_smoke_trace_bound',latest is not None)
        if latest:
            applied=sum(e['applied'] for e in latest)
            check.need('latest_smoke_counters',smoke['metrics']['optimizer_steps_applied']==applied and smoke['metrics']['optimizer_steps_skipped']==50-applied)
            check.need('smoke_stability_policy',decision['passed'] is True and (50-applied)/50<=.05)
        log=(folder/'commands.log').read_text(encoding='utf-8')
        commands=[json.loads(line[len('COMMAND '):]) for line in log.splitlines() if line.startswith('COMMAND ')]
        training=[c for c in commands if 'src.training.train_crossdomain' in c and '--benchmark_steps' not in c]
        check.need('fresh_full_training_command',len(training)==1 and '--resume_from' not in training[0] and '--max_steps' not in training[0] and '--dry_run' not in training[0])
        check.need('zero_step_identity_log','Max Logit Diff = 0.0000000000e+00' in log)
        check.need('no_fatal_log',not re.search(r'Traceback|OutOfMemoryError|\[X\] SU CO',log))
        references[arm]=dict(path=cfg['training']['backup_dir']+'/best.pth',sha256=chosen['sha256'],locally_supplied=(folder/'best.pth').exists(),
            evidence='GPU run_verification reported this hash; local bytes were not supplied, so no local model reload/forward was performed.')
        summaries[arm]=dict(selected=chosen,training_events=len(trace),optimizer_applied=sum(e['applied'] for e in trace),
            optimizer_skipped=sum(not e['applied'] for e in trace),nonfinite_grad_events=sum(not e['grad_norm_finite'] for e in trace),
            applied_nonfinite=0,epoch_diagnostics=per_epoch,smokes=smoke_traces,
            epoch_compute_minutes=sum(float(r['time_sec']) for r in history)/60,
            selected_correct_counts={key:int(round(float(selected[key])*denom)) for key,denom in [('clean_r',590),('clean_f',490),('q30_r',150),('q30_f',150)]},
            smoke_ms_per_minibatch=smoke['ms_per_step'],smoke_peak_allocated_gib=smoke['peak_vram_gib'])
    context_diff={k:[contexts[ARMS[0]].get(k),contexts[ARMS[1]].get(k)] for k in set(contexts[ARMS[0]])|set(contexts[ARMS[1]]) if contexts[ARMS[0]].get(k)!=contexts[ARMS[1]].get(k)}
    check.need('controlled_context_difference',context_diff=={'preprocessing':['legacy_imagenet','clip']})
    normalized=[]
    for arm in ARMS:
        config=copy.deepcopy(configs[arm]);config['experiment'].pop('name');config['dataset'].pop('preprocessing')
        for key in ['save_dir','backup_dir','history_csv']:config['training'].pop(key)
        normalized.append(config)
    check.need('controlled_config_difference',normalized[0]==normalized[1])
    check.need('epoch_zero_validation_identical',rows(source/ARMS[0]/'val_groups.csv')[0]==rows(source/ARMS[1]/'val_groups.csv')[0])
    winner=max(ARMS,key=lambda arm:summaries[arm]['selected']['worst_group'])
    check.need('winner_recomputed',selection['selected']==winner)
    check.need('validation_only_claim',selection['selection_split']=='val_select' and selection['quality_claim'].startswith('validation-only'))
    check.need('completion_not_gate_pass',status['status']=='training_completed_validation_selected' and status['quality_gate_pass'] is None and status['source_commit']==commit)
    check.need('single_run_id',len(run_ids)==1)
    p0,p1=(summaries[a]['selected'] for a in ARMS)
    differences={key:100*(p0[key]-p1[key]) for key in ['clean_r','clean_f','q30_r','q30_f','worst_group']}
    check.finish()
    return dict(schema='phase7-training-pilot-local-audit/v1',passed=True,run_id=next(iter(run_ids)),source_commit=commit,
        source_fingerprint=fingerprint,source_files_checked=len(hashes),source_code_sha256=code_hash,notebook=notebook_summary,
        source_file_hashes=hashes,
        GPU=status['GPU'],arms=summaries,selected=winner,control_minus_clip_pp=differences,context_differences=context_diff,
        model_references=references,checks=check.groups,data_checks=sum(v['checked'] for v in check.groups.values()),
        limitations=['Weights were not supplied; reported checkpoint hashes were not independently rehashed locally.',
            'Validation-only, one training seed, no per-image predictions for a paired uncertainty estimate.',
            'The 300 Q30 records are degraded versions from val_select, not 300 additional independent originals.',
            'One more AMP update was skipped in p1; equal attempted steps do not mean exactly equal applied steps.',
            'Current deployment checkpoint and G7.2/G7.3 are unchanged; no new CFE/in-dist/packaging test is present.'])


def report_markdown(result):
    def pct(value):return f'{100*value:.2f}'.replace('.',',')
    p0,p1=(result['arms'][arm] for arm in ARMS)
    selected=result['selected'];reference=result['model_references'][selected]
    table=['| Chỉ số trên val_select | P0: train legacy | P1: train CLIP |',
           '| :--- | ---: | ---: |']
    for label,key in [('Clean Real ACC','clean_r'),('Clean Fake ACC','clean_f'),('Q30 Real ACC','q30_r'),('Q30 Fake ACC','q30_f'),('Worst-group','worst_group')]:
        table.append(f"| {label} | {pct(p0['selected'][key])}% | {pct(p1['selected'][key])}% |")
    return f'''# Hậu kiểm pilot {result['run_id']}

Ngày tổng hợp: 10/11/2026 (Asia/Saigon). **Hai nhánh đã hoàn tất 5 epoch trên NVIDIA L4; chọn `{selected}` theo validation. Chưa có kết quả chất lượng cuối cùng cho checkpoint mới.**

## 1. Tính toàn vẹn và phạm vi xác minh

- `raw/`: 27 file nộp từ `wait`, giữ nguyên byte, gồm cả `commands.log` và các trace smoke cũ/mới.
- `evidence/executed_notebook.ipynb`: bản sao notebook đã chạy; 4 cell có execution 1–4, không có error output.
- Commit source: `{result['source_commit']}`. Cả {result['source_files_checked']} hash file khớp byte Git trên Linux; fingerprint `{result['source_fingerprint']}`. Khi hậu kiểm trên Windows đã tắt phép chuyển EOL của `git archive`, không chỉnh source để ép hash.
- **{result['data_checks']:,} phép đối chiếu dữ liệu đạt**, không phải số lượng unit test. Xem `analysis/verification.json` và `import_manifest.json`.
- Preflight GPU báo ảnh staged/audit/split/canonical PASS. Local kiểm tra ràng buộc hash và log; không có ảnh staged hoặc weights mới trong gói để tự chạy lại forward.

## 2. So sánh theo thiết kế đã chốt

Hai nhánh cùng seed 42, 5 epoch, 765 mini-batch/epoch, batch 16 × accumulation 2, cùng manifest/khởi tạo/teacher/loss/LR. Cả hai **validation CLIP**, 1.080 ảnh clean (590 Real, 490 Fake), cache Q30 300 ảnh (150/150), T=1,6402 và ngưỡng=0,9127 đóng băng. Chỉ training preprocessing khác nhau; epoch 0 giống hệt nhau.

{chr(10).join(table)}

Cả hai chọn epoch 5 đúng luật worst-group với Clean Real ≥90%. Worst-group ở đây là min bốn recall clean/Q30 × Real/Fake, không phải worst-subset của CFE.

P0 hơn P1 **{result['control_minus_clip_pp']['worst_group']:.2f} điểm phần trăm** worst-group: Q30 Fake đúng **63/150 so với 53/150**, thêm 10 ảnh. Clean Fake đúng **296/490 so với 264/490**, hơn {result['control_minus_clip_pp']['clean_f']:.2f} điểm phần trăm. P1 giữ Real tốt hơn: **589/590 so với 586/590**; không có thắng tuyệt đối ở mọi chỉ số.

**Kết luận công tâm:** giả thuyết đổi training preprocessing sang CLIP chưa được kết quả seed này hỗ trợ; giữ P0 làm ứng viên cho bước đo tiếp. Không khẳng định CLIP luôn kém hoặc chênh lệch có ý nghĩa thống kê: chỉ một seed, không có dự đoán từng ảnh để ước lượng paired CI. Các ảnh Q30 là biến thể từ val_select, không phải 300 ảnh gốc độc lập bổ sung. Validation được xem qua nhiều epoch nên điểm checkpoint đã chọn có độ lạc quan do selection.

## 3. Gradient và cập nhật optimizer

- P0: **{p0['optimizer_applied']}/{p0['training_events']}** lần optimizer thực sự được gọi; bỏ qua {p0['optimizer_skipped']} lần, tại epoch 1/minibatch 4. AMP scale 2048→1024.
- P1: **{p1['optimizer_applied']}/{p1['training_events']}** lần; bỏ qua {p1['optimizer_skipped']} lần, tại epoch 1/minibatch 4 và epoch 4/minibatch 102. AMP scale 2048→1024→512.
- Tất cả ba norm không hữu hạn được AMP bỏ qua; **không có cập nhật áp dụng norm không hữu hạn**, không có epoch toàn bộ update bị bỏ qua. CSV gradient là trung bình các norm hữu hạn, được tính lại từ trace.
- Hai nhánh có cùng 1.915 lượt thử, nhưng số update thực khác nhau một lần; đây là giới hạn cần ghi nhận. Không được mô tả gradient luôn hữu hạn hoặc cả hai có số update thực bằng nhau.
- Có hai vòng smoke cho mỗi nhánh (cell smoke riêng và kiểm tra lại trước train). Từng vòng có 3 update warmup + 50 update phần đo. Ở phần đo cuối: P0 skip 0/50, P1 skip 1/50=2%, đều đạt chính sách kỹ thuật ≤5%; warmup được lưu riêng.

Trace mới giải thích được overflow/skip của **run này**. Không suy ngược rằng mọi NaN trong run huấn luyện cũ chắc chắn cùng nguyên nhân.

## 4. Thời gian và bộ nhớ

Smoke L4: P0 {p0['smoke_ms_per_minibatch']:.2f} ms/mini-batch, P1 {p1['smoke_ms_per_minibatch']:.2f} ms/mini-batch; peak allocated khoảng {p0['smoke_peak_allocated_gib']:.2f} GiB. Đây là thời gian huấn luyện gồm backward, không phải inference latency.

Tổng thời gian các epoch được CSV ghi: P0 {p0['epoch_compute_minutes']:.2f} phút, P1 {p1['epoch_compute_minutes']:.2f} phút. Metadata notebook ghi cell smoke {result['notebook']['smoke_wall_minutes']:.2f} phút, cell train {result['notebook']['train_wall_minutes']:.2f} phút; hai phạm vi khác nhau vì cell còn gồm preflight/smoke lặp lại/validation/backup. Không dùng tổng epoch time làm tổng wall time phiên Colab.

## 5. Model được chọn và giới hạn bàn giao

Ứng viên: `{selected}`, epoch 5. File Drive:

```text
{reference['path'].removeprefix('/content/drive/')}
```

SHA-256 được GPU report ghi:

```text
{reference['sha256']}
```

`run_verification.json` ràng buộc hash model/log tại GPU; local xác minh được hash CSV và sự nhất quán metadata/selection. ZIP không có `.pth`, vì vậy **chưa tự băm lại hoặc nạp model mới tại local**. Xem `candidate_reference.json` và `analysis/model_references.json`.

Chưa có logits/probabilities test, AUC/AP/ECE/FPR cho model này, chưa có CFE 1.600/18 tập in-dist hoặc benchmark đóng gói. Không so trực tiếp worst-group 42% ở đây với 46,67% của run cũ vì validation preprocessing đã khác. Checkpoint deployment hiện hành và phán quyết G7.2 FAIL 6/8, G7.3 pending vẫn giữ nguyên.

## 6. Việc tiếp theo

1. Khóa P0/hash trên như ứng viên theo validation; không cần train lại chỉ để sửa báo cáo của vòng này.
2. Trên Colab, kiểm tra hash và canonical load chính file P0 `best.pth`; trích logits cho đúng 400 calibration đã đóng băng, fit T/ngưỡng bằng bộ tối ưu có kiểm tra hội tụ. T/ngưỡng lựa chọn ở trên không phải calibration cuối của model mới.
3. Đánh giá P0 và model deployment hiện hành bằng protocol phù hợp để so sánh: cùng IDs, tiền xử lý, batch/precision/backend; lưu dự đoán từng ảnh. Chạy CFE/18 tập và các gate, không chọn lại model bằng test.
4. Tạo gói đo/đóng gói mới gắn hash P0. Notebook `phase7_corrected_rerun_colab.ipynb` cũ khóa source `5cd49252…`, không dùng nguyên nó để chấm weights mới. Chỉ thay model hiện hành sau khi có bằng chứng tương ứng; nếu cần kết luận tổng quát, bổ sung seed/test chưa xem theo kế hoạch riêng.

Gói này đủ để chốt **pilot training đã hoàn tất và đã hậu kiểm log**, chưa đủ để gọi toàn bộ Phase 7 PASS hoặc tuyên bố model mới tốt hơn incumbent trên test.
'''


def import_submission(source,notebook,output_root=None):
    source,notebook=Path(source).resolve(),Path(notebook).resolve()
    notebook_before=sha(notebook)
    protected=read(ROOT/'DATASET/logs/phase7/task_7.16/documentation_manifest.json')['protected']
    before={name:sha(ROOT/name) for name in protected}
    if before!=protected:raise ValueError('Historical protected inputs differ before import; inspect before proceeding')
    result=verify_submission(source,notebook)
    destination=(Path(output_root).resolve() if output_root else ROOT/'DATASET/logs/phase7/improvement_round1/runs')/result['run_id']
    if not destination.is_relative_to(ROOT):raise ValueError('Import destination must stay inside this repository')
    manifest=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        if path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('Source links may not escape the submitted folder')
        relative=path.relative_to(source);target=destination/'raw'/relative
        expected=sha(path)
        if target.exists() and sha(target)!=expected:raise ValueError('Existing imported bytes differ; do not overwrite a run')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        if sha(target)!=expected:raise IOError('Copied raw file checksum mismatch')
        manifest.append(dict(path=relative.as_posix(),bytes=path.stat().st_size,sha256=expected))
    evidence=destination/'evidence/executed_notebook.ipynb';evidence.parent.mkdir(parents=True,exist_ok=True)
    if evidence.exists() and sha(evidence)!=notebook_before:raise ValueError('Existing notebook evidence differs; preserve the earlier evidence')
    if not evidence.exists():shutil.copy2(notebook,evidence)
    if sha(evidence)!=notebook_before:raise IOError('Notebook evidence checksum mismatch')
    write(destination/'analysis/verification.json',result)
    write(destination/'analysis/model_references.json',result['model_references'])
    selected=result['selected'];reference=result['model_references'][selected]
    write(destination/'candidate_reference.json',dict(schema='phase7-pilot-candidate-reference/v1',run_id=result['run_id'],
        arm=selected,epoch=result['arms'][selected]['selected']['epoch'],**reference,
        state='validation_selected; calibration and locked benchmark pending; not a deployment replacement',
        deployed_incumbent='DATASET/logs/phase7/task_7.15/current',
        required_next_steps=['Verify actual weight bytes and canonical load','Fit checkpoint-specific calibration on frozen 400 IDs',
                             'Evaluate frozen 1600 IDs and 18 in-dist sets','Matched incumbent/best-reference comparison','Package/verify same new checkpoint']))
    with (destination/'analysis/comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['arm','selected_epoch','clean_real_pct','clean_fake_pct','q30_real_pct','q30_fake_pct','worst_group_pct','optimizer_applied','optimizer_skipped','sha256']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for arm in ARMS:
            data=result['arms'][arm];point=data['selected']
            writer.writerow(dict(arm=arm,selected_epoch=point['epoch'],clean_real_pct=100*point['clean_r'],clean_fake_pct=100*point['clean_f'],
                q30_real_pct=100*point['q30_r'],q30_fake_pct=100*point['q30_f'],worst_group_pct=100*point['worst_group'],
                optimizer_applied=data['optimizer_applied'],optimizer_skipped=data['optimizer_skipped'],sha256=point['sha256']))
    with (destination/'analysis/epoch_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm',*result['arms'][ARMS[0]]['epoch_diagnostics'][0]])
        writer.writeheader()
        for arm in ARMS:
            writer.writerows(dict(arm=arm,**row) for row in result['arms'][arm]['epoch_diagnostics'])
    (destination/'README.md').write_text(report_markdown(result),encoding='utf-8')
    if sha(notebook)!=notebook_before or any(sha(ROOT/name)!=expected for name,expected in before.items()):
        raise ValueError('Protected notebook/history changed during import')
    write(destination/'import_manifest.json',dict(schema='phase7-pilot-import/v1',run_id=result['run_id'],
        source=source.relative_to(ROOT).as_posix(),raw_files=manifest,raw_file_count=len(manifest),raw_bytes=sum(row['bytes'] for row in manifest),
        notebook=dict(path=notebook.relative_to(ROOT).as_posix(),sha256=notebook_before,unchanged=True),
        old_protected_files_verified=len(protected),old_notebooks_preserved=32,old_raw_files_preserved=55,
        weights_supplied=any(path.suffix=='.pth' for path in source.rglob('*') if path.is_file()),
        imported_at_utc=datetime.now(timezone.utc).isoformat(),scientific_scope='validation-only, single seed'))
    base=ROOT/'DATASET/logs/phase7/improvement_round1'
    write(base/'index.json',dict(schema='phase7-pilot-index/v1',current_run=result['run_id'],
        path=destination.relative_to(ROOT).as_posix(),state='training_completed_logs_audited',
        selected=selected,selected_sha256=reference['sha256'],final_model_replaced=False,
        G72='unchanged: FAIL 6/8',G73='unchanged: pending matched reference'))
    print(json.dumps(dict(passed=True,run_id=result['run_id'],data_checks=result['data_checks'],
        imported_raw_files=len(manifest),selected=selected,destination=destination.relative_to(ROOT).as_posix()),ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'wait/phase7_improvement_round1/pilot_results_without_weights')
    parser.add_argument('--notebook',type=Path,default=ROOT/'notebooks/task/phase7_improvement_round1/phase7_improvement_round1_colab.ipynb')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        result=verify_submission(args.source,args.notebook)
        print(json.dumps(dict(passed=result['passed'],run_id=result['run_id'],data_checks=result['data_checks'],selected=result['selected'])))
    else:import_submission(args.source,args.notebook)


if __name__=='__main__':main()
