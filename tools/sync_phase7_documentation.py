"""Build traceable Phase 7 documentation without changing experiment artifacts.

Run --catalog-only before --write. Run --check to verify the written documents.
Human sign-off and experimental gates are deliberately separate from doc checks.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'DATASET/logs/phase7/task_7.16'
RUN = 'DATASET/logs/phase7/task_7.15/current'
EXPECTED_RUN = 'corrected_20261109_102954_3d28ad'
EXPECTED_PACKAGE = '248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0'
START = '<!-- PHASE7_CURRENT_BEGIN -->'
END = '<!-- PHASE7_CURRENT_END -->'
TARGETS = [
    'docs/plan/ke_hoach.md',
    'docs/plan/ke_hoach_cac_cot_moc_va_diem_dung.md',
    'docs/plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md',
    'docs/bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md',
    'docs/plan/ke_hoach_test_communityforensics_eval.md',
    'docs/tasks/member_a_data_infra/README_A.md',
    'docs/tasks/member_b_architecture_loss/README_B.md',
    'docs/tasks/member_c_training_eval/README_C.md',
    'docs/bao_cao/overall/README.md',
    'docs/bao_cao/overall/tong_quan_du_an.md',
    'docs/bao_cao/C/bao_cao_task_6.6_r_locked_recovery.md',
]
OUTPUTS = [
    'docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md',
    'docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md',
    'docs/bao_cao/overall/slide_phase7_5phut.md',
    'docs/milestones/milestone_phase7_acceptance.md',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def pointer(document, path):
    for token in path.strip('/').split('/') if path else []:
        token = token.replace('~1', '/').replace('~0', '~')
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def f(value, places=2):
    return format(float(value), f'.{places}f').replace('.', ',')


def count(value):
    return format(int(value), ',').replace(',', '.')


def link(owner, target, label=None):
    import os
    relative = os.path.relpath(ROOT / target, (ROOT / owner).parent).replace('\\', '/')
    return f'[{label or Path(target).name}]({relative})'


def strict_sites():
    result = []
    for folder in ['src', 'tools']:
        for path in sorted((ROOT / folder).rglob('*.py')):
            try:
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            except (SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != 'strict':
                        continue
                    value = keyword.value
                    if isinstance(value, ast.Constant) and value.value is True:
                        continue
                    relative = path.relative_to(ROOT).as_posix()
                    expression = ast.unparse(value)
                    if relative == 'src/models/canonical_loader.py':
                        scope = 'canonical: allow_partial explicit; final path uses default false'
                    elif relative == 'tools/eval_gen1_reinterpretation.py':
                        scope = 'forensic diagnostic only; invalid_layout; gate-ineligible'
                    elif relative.startswith('tools/test_'):
                        scope = 'initialization/test; not current measured-report path'
                    elif relative == 'src/models/clip/model.py':
                        scope = 'base CLIP initialization; final checkpoint load validated later'
                    else:
                        scope = 'legacy/dynamic path; not permission to report final metrics'
                    result.append(dict(file=relative, line=node.lineno, call=ast.unparse(node.func), strict=expression, scope=scope))
    return result


def build_catalog():
    LOG.mkdir(parents=True, exist_ok=True)
    paths = {
        'benchmark': RUN + '/raw/benchmark/final_report.json',
        'acceptance': RUN + '/raw/acceptance.json',
        'calibration': RUN + '/calibration/anchored_v2.json',
        'latency': RUN + '/raw/latency_phase7.json',
        'verification': RUN + '/raw/real_image_verification.json',
        'audit': RUN + '/analysis/verification.json',
        'recomputed': RUN + '/analysis/recomputed_metrics.json',
        'distribution': RUN + '/analysis/distribution_cautions.json',
        'ci': RUN + '/analysis/ci_verification.json',
        'canonical': 'DATASET/logs/phase7/wise_ft_sweep/task_7.2_canonical_load_verification.json',
        'structure': 'DATASET/logs/phase7/wise_ft_sweep/checkpoint_structure_audit.json',
        'gen1': 'DATASET/logs/phase7/wise_ft_sweep/wise_ft_legacy_reinterpretation.json',
        'matrix': 'DATASET/logs/phase7/wise_ft_sweep/cfe_v3_baseline_matrix.json',
        'data': 'DATASET/logs/phase7/data_isolation_report.json',
        'manifest': 'DATASET/manifests/data_manifest_v2.json',
        'degradation': 'DATASET/logs/phase7/degradation_v2/degradation_benchmark.json',
        'selection': 'DATASET/logs/phase7/selection_benchmark.json',
        'router': 'DATASET/logs/phase7/router_params.json',
        'ablation': 'DATASET/logs/phase7/ablation_v2_fixed_run1/ablation_mini_v2.json',
    }
    sources = {key: read(path) for key, path in paths.items()}
    b, a, c, l, v = (sources[k] for k in ['benchmark', 'acceptance', 'calibration', 'latency', 'verification'])
    package = b['package_sha256']
    assert package == EXPECTED_PACKAGE and sources['audit']['run_id'] == EXPECTED_RUN, 'New runs require a new evidence review before generating conclusions'
    assert package == a['package_sha256'] == l['checkpoint_sha256'] == v['checkpoint_sha256']
    assert b['calibration']['calibration_sha256'] == digest(ROOT / paths['calibration'])
    assert b['gates']['G7.2'] == a['G72']
    assert a['G72']['pass'] is False and a['G73']['status'] == 'pending'
    assert sum(item['pass'] for item in a['G72']['criteria'].values()) == 6
    assert sources['audit']['package_sha256'] == package
    xml_path = LOG / 'validation/targeted_tests.xml'
    tests = ET.parse(xml_path).getroot()
    cases = list(tests.iter('testcase'))
    test_record = dict(cases=len(cases), failures=len(list(tests.iter('failure'))),
                       errors=len(list(tests.iter('error'))), skipped=len(list(tests.iter('skipped'))),
                       source=xml_path.relative_to(ROOT).as_posix(), sha256=digest(xml_path),
                       scope='canonical_loader, calibration_partitions, temperature_convergence, phase7_acceptance, router_pipeline; not full repository pytest')
    assert test_record['cases'] == 103 and not test_record['failures'] and not test_record['errors'] and not test_record['skipped']
    write_json(LOG / 'validation/targeted_tests.json', test_record)
    component_xml = LOG / 'validation/component_tests.xml'
    component_tree = ET.parse(component_xml).getroot()
    component_record = dict(cases=len(list(component_tree.iter('testcase'))),
        failures=len(list(component_tree.iter('failure'))), errors=len(list(component_tree.iter('error'))),
        skipped=len(list(component_tree.iter('skipped'))), source=component_xml.relative_to(ROOT).as_posix(),
        sha256=digest(component_xml), scope='loss_v2, selection_metric, selection_regressions, degradation_v2; disjoint test files from the 103-case run')
    assert component_record['cases'] == 53 and not component_record['failures'] and not component_record['errors'] and not component_record['skipped']
    write_json(LOG / 'validation/component_tests.json', component_record)
    test_record['additional_components'] = component_record
    test_record['total_disjoint_cases'] = test_record['cases'] + component_record['cases']
    write_json(LOG / 'validation/targeted_tests.json', test_record)

    # Read the actual selected file, not a stale sidecar for an unselected candidate.
    import torch
    gen2_path = ROOT / 'DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth'
    container = torch.load(gen2_path, map_location='cpu', weights_only=True, mmap=True)
    embedded = container['metadata']
    gen2_sha = digest(gen2_path)
    rows = list(csv.DictReader((ROOT / 'DATASET/logs/phase7/wise_ft_sweep/wise_ft_v2_sweep.csv').open(encoding='utf-8-sig')))
    selected = [r for r in rows if r['selected'].lower() == 'true']
    assert len(rows) == 12 and len(selected) == 1
    row = selected[0]
    assert float(row['alpha']) == embedded['alpha'] == .55
    assert float(row['beta_srm']) == embedded['beta_srm'] == .35
    assert gen2_sha == next(r['sha256'] for r in sources['matrix']['checks'] if r['model'] == 'wise_ft_v2')
    stale = read('DATASET/logs/phase7/wise_ft_sweep/wise_ft_v2_best_metadata.json')
    gen2_record = dict(schema='gen2_documentation_verification/v1', path=gen2_path.relative_to(ROOT).as_posix(),
                       sha256=gen2_sha, metadata=embedded, sweep_selected=row,
                       stale_sidecar='DATASET/logs/phase7/wise_ft_sweep/wise_ft_v2_best_metadata.json',
                       stale_sidecar_sha256=stale['sha256'], stale_sidecar_selection_status=stale['selection_status'],
                       note='Sidecar alpha=.50 is an unselected older candidate. Embedded metadata, actual bytes, selected CSV and measured matrix agree on alpha=.55/beta=.35.')
    write_json(LOG / 'gen2_verified_metadata.json', gen2_record)
    del container

    history = list(csv.DictReader((ROOT / 'DATASET/logs/phase7/anchored_v2_run1/train_history.csv').open(encoding='utf-8-sig')))
    best = max(history, key=lambda r: float(r['worst_group']))
    train_record = dict(epochs=len(history), selected_epoch=int(best['epoch']),
                        selected_worst_group_pct=100*float(best['worst_group']),
                        selected_clean_real_pct=100*float(best['clean_r']),
                        nonfinite_grad_norm_epochs=[int(r['epoch']) for r in history if not math.isfinite(float(r['grad_norm']))],
                        source='DATASET/logs/phase7/anchored_v2_run1/train_history.csv',
                        source_sha256=digest(ROOT / 'DATASET/logs/phase7/anchored_v2_run1/train_history.csv'),
                        note='NaN telemetry is unresolved; finite current predictions do not prove all training gradients were finite.')
    write_json(LOG / 'training_summary.json', train_record)
    sites = strict_sites()
    write_json(LOG / 'strict_loading_review.json', dict(schema='strict_loading_scope_review/v1', sites=sites,
               repository_wide_clean=False, current_final_loader='src/models/canonical_loader.py',
               conclusion='Current report path is canonical; broad legacy cleanup/exception audit remains open. No source loader was changed by Task 7.16.'))
    summary = dict(schema='phase7_documentation_summary/v1', report_schema='v3', derived_after_run=True,
        source_runtime_schema=b['schema'], source_acceptance_schema=a['schema'], run_id=sources['audit']['run_id'],
        scientific_status='post_hoc_confirmation; not a new GPU run', package_sha256=package,
        source_checkpoint_sha256=b['source_checkpoint_sha256'], environment=b['environment'],
        calibration=c, cfe=b['cfe'], indist=b['indist'], gates={
            'G7.0':dict(status='pending_repo_wide_legacy_cleanup', canonical_core='verified', gen1='rejected_invalid_layout',
                         note='Literal grep and partial diagnostic CI requirements are not fully closed repository-wide.'),
            'G7.1':dict(status='FAIL', source_verdict=sources['matrix']['gate_verdict'],
                         selected_clean_fake_at_real90=sources['matrix']['results']['wise_ft_v2']['clean']['fake_at_real90'],
                         selected_clean_auc=sources['matrix']['results']['wise_ft_v2']['clean']['auc'],
                         selected_q30_auc=sources['matrix']['results']['wise_ft_v2']['q30']['auc'],
                         note='The old matrix threshold key applies 55 to Q30 Fake@Real90; master-plan clean criterion also fails. Do not silently relabel the raw report.'),
            'G7.2':a['G72'], 'G7.3':a['G73']}, latency=l, real_image_verification=v,
        source_reports={paths[k]:digest(ROOT/paths[k]) for k in ['benchmark','acceptance','calibration','latency','verification','audit']},
        gaps_pp={k:float(item['target'].split('>=')[1].split('%')[0])-float(item['value'])
                 for k,item in a['G72']['criteria'].items() if not item['pass']},
        local_new_model_weights_available=False, human_signoff='pending; no signatures generated',
        data_v2_audit_scope='14,400 source records; distinct from the 400/1,600 CFE operating partition',
        bootstrap=b['bootstrap'], distribution_cautions=sources['distribution'],
        recomputed_metrics=sources['recomputed'])
    write_json(LOG / 'final_phase7_report.json', summary)

    facts = []
    def add(key, file, ptr, unit='', scope='current'):
        value = pointer(read(file), ptr)
        facts.append(dict(id=key, file=file, pointer=ptr, value=value, unit=unit, scope=scope))
    for condition in ['clean','q30']:
        for key in ['auc','ap','fake_at_real90','accuracy','real_acc','fake_acc','ece','nll','ci95','tpr_at_fpr']:
            add(f'cfe.{condition}.{key}',paths['benchmark'],f'/cfe/{condition}/metrics/{key}','percent except nll')
    for key in ['macro_acc','pooled_real_acc','mean_subset_real_acc']:
        add('indist.'+key,paths['benchmark'],'/indist/'+key,'percent')
    add('fixed_fpr.q30',paths['recomputed'],'/pooled/q30/fixed_fpr_pct','percent')
    add('distribution.macro_ci',paths['distribution'],'/macro_subset_ci95')
    add('distribution.boundary_cases',paths['distribution'],'/cpu_gpu_boundary_rows_by_dataset')
    add('bootstrap',paths['benchmark'],'/bootstrap')
    for key in ['T','T_star_default','n_calib','fpr_target','fpr_measured','ece_after','nll_after']:
        add('calibration.'+key,paths['calibration'],'/'+key)
    for key in ['overhead_delta_ms','package_per_image_ms','overhead_delta_p95_ms']:
        add('latency.batch1.'+key,paths['latency'],'/batches/1/'+key,'ms/image')
    for name in ['data','degradation','selection','router','gen1']:
        keys={'data':['/integrity/summary','/elapsed_s','/violations','/pass'],
              'degradation':['/pct_overhead','/source','/passed'],
              'selection':['/ratio','/model','/passed'],
              'router':['/status','/deployment_enabled','/experiment_status'],
              'gen1':['/runs','/delta_fake_acc','/verdict']}[name]
        for ptr in keys:add(name+'.'+ptr.strip('/').replace('/','.'),paths[name],ptr,scope='historical diagnostic' if name=='gen1' else 'recorded artifact')
    for name,ptr in [('gen2','/metadata'),('training','/selected_worst_group_pct'),('training','/nonfinite_grad_norm_epochs'),('tests','/cases'),('tests','/total_disjoint_cases')]:
        file={'gen2':'gen2_verified_metadata.json','training':'training_summary.json','tests':'validation/targeted_tests.json'}[name]
        add(name+'.'+ptr.strip('/'),'DATASET/logs/phase7/task_7.16/'+file,ptr)
    catalog = dict(schema='phase7_documentation_evidence/v1',run_id=summary['run_id'],facts=facts,
        sources={name:dict(path=path,sha256=digest(ROOT/path)) for name,path in paths.items()},
        data_download_bytes_reported=sum(item['bytes_downloaded'] for item in sources['manifest']['metadata']['provenance']),
        data_stream_total_elapsed_s=None, data_stream_note='Not recorded; audit elapsed_s is not stream time.',
        protected_notebooks={p.relative_to(ROOT).as_posix():digest(p) for p in ROOT.glob('notebooks/task/task_7.1[45]/*.ipynb')},
        source_original_notebook_sha256=digest(ROOT/'notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb'),
        source_script_sha256=digest(Path(__file__)))
    write_json(LOG / 'evidence_catalog.json',catalog)
    with (LOG/'source_values.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','file','pointer','value','unit','scope']);writer.writeheader()
        writer.writerows({**item,'value':json.dumps(item['value'],ensure_ascii=False)} for item in facts)
    return sources,summary,catalog,gen2_record,train_record,test_record,sites


def source_map_report(data):
    s,summary,catalog,gen2,train,tests,sites=data
    path='docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md'
    b=s['benchmark'];c=s['calibration'];rows=[]
    mapping=[
      ('52.05%','Gen 1 n=8, ACC diagnostic, invalid_layout','CFE clean accuracy',f(b['cfe']['clean']['metrics']['accuracy'])+'%',catalog['sources']['benchmark']['path'],'/cfe/clean/metrics/accuracy','Không phải mức tăng cùng protocol; model và partition khác nhau.'),
      ('4.10%','Gen 1 n=8, Fake ACC diagnostic, invalid_layout','CFE clean Fake ACC',f(b['cfe']['clean']['metrics']['fake_acc'])+'%',catalog['sources']['benchmark']['path'],'/cfe/clean/metrics/fake_acc','Chỉ ở ngưỡng vận hành hiện hành; không thay thế Fake@Real90.'),
      ('96.58%','Overall Gen 1 bị invalid_layout, từng gọi sai là macro','In-dist mean 18 balanced ACC',f(b['indist']['macro_acc'])+'%',catalog['sources']['benchmark']['path'],'/indist/macro_acc','Không so trực tiếp overall cũ với macro mới.'),
      ('91.5%','Mốc Phase 6-R, chỉ để tham khảo lịch sử','CFE clean AUC',f(b['cfe']['clean']['metrics']['auc'])+'%',catalog['sources']['benchmark']['path'],'/cfe/clean/metrics/auc','Không đổi ACC thành AUC; chỉ thay nguồn trích dẫn hiện hành.'),
      ('5/8 G7.2','Run operating point baseline trước sửa','G7.2 hiện hành','6/8; FAIL',catalog['sources']['acceptance']['path'],'/G72','Điểm ECE Q30 nay đạt; hai mục vẫn FAIL.'),
      ('13.49% ECE Q30','Run baseline operating point','ECE Q30',f(b['cfe']['q30']['metrics']['ece'])+'%',catalog['sources']['benchmark']['path'],'/cfe/q30/metrics/ece','Khác calibration/precision; không gán là tăng năng lực phân biệt.'),
      ('T=1.4676 / 2.78636','Bản trước hội tụ / local FP32','T FP16/CUDA batch16',str(c['T']),catalog['sources']['calibration']['path'],'/T','Không ép T bằng cache local; ngưỡng cũng fit lại.'),
      ('0.9127 / 0.795321','Ngưỡng baseline / local FP32','Ngưỡng hiện hành',str(c['T_star_default']),catalog['sources']['calibration']['path'],'/T_star_default','Fit trên 400 calibration, không dùng evaluation.'),
      ('44.79 ms hoặc 44.98 ms','Backbone/package tổng, từng nhầm với overhead','Overhead batch1',f(s['latency']['batches']['1']['overhead_delta_ms'],6)+' ms',catalog['sources']['latency']['path'],'/batches/1/overhead_delta_ms','Median từng cặp đo; không lấy hiệu hai median đã làm tròn.'),
      ('Chờ 20 ảnh / chờ T4','Trạng thái trước run GPU','Verification','20 ảnh; L4 đã đo',catalog['sources']['verification']['path'],'/n_images','T4 không bắt buộc theo yêu cầu người dùng.'),
      ('Router thắng / 71 tests = nghiệm thu','Số synthetic v1 đã thu hồi','Router','Tắt; experiment deferred',catalog['sources']['router']['path'],'/deployment_enabled','Unit test không thay cho thí nghiệm held-out.'),
      ('Gen 2 alpha=.50 metadata rời','Ứng viên cũ, unselected','Gen 2 đã chọn','alpha=.55; beta=.35','DATASET/logs/phase7/task_7.16/gen2_verified_metadata.json','/metadata','Đã băm actual .pth và đối chiếu metadata nhúng + CSV + matrix.'),
      ('12–16k dự kiến','Kho dữ liệu v2','Số bản ghi','14.400',catalog['sources']['data']['path'],'/integrity/summary/n_records','Train 12.240; val_calib 1.080; val_select 1.080, khác CFE 400/1.600.'),
    ]
    for old,where,label,new,file,ptr,note in mapping:
        rows.append(f'| {old} | {where} | {label}: **{new}** | {link(path,file)} `{ptr}` | {note} |')
    body=f'''# Rà soát số liệu Phase 7 — nguồn cho Task 7.16

Run hiện hành `{summary['run_id']}`; package `{summary['package_sha256']}`. Tài liệu này được lập trước khi cập nhật 11 tệp mục tiêu. Chỉ các số mới có file và JSON pointer bên dưới được dùng cho báo cáo hiện hành.

## Số cũ → số mới → nguồn

Các số cũ trong bảng chỉ để tham khảo lịch sử; Gen 1 là `invalid_layout`. Không hiểu các cặp số khác metric/protocol là mức cải thiện trực tiếp.

| Số/mô tả cũ | Ngữ cảnh cũ | Hiện hành | File + trường JSON | Diễn giải |
| :--- | :--- | :--- | :--- | :--- |
'''+ '\n'.join(rows)+f'''

Toàn bộ giá trị chi tiết, kể cả AUC/AP/TPR/CI, count, calibration và latency: {link(path,'DATASET/logs/phase7/task_7.16/evidence_catalog.json')} và {link(path,'DATASET/logs/phase7/task_7.16/source_values.csv')}. Hash nguồn được giữ trong catalog; không gõ số slide độc lập.

## Những điều không được suy ra

G7.0 còn DoD legacy trên toàn repo; G7.1 FAIL; G7.2 FAIL 6/8; G7.3 pending reference tốt nhất theo điều kiện. Không biến việc hoàn tất thí nghiệm hoặc viết tài liệu thành PASS tất cả cổng.

Pooled accuracy, macro balanced accuracy và mean-subset Real ACC là ba đại lượng khác nhau. FPR calibration 5% không bảo đảm FPR của miền khác. CI theo ảnh không bao quát phụ thuộc giữa generator/dataset. Full checkpoint GPU không có trong ZIP; không dùng file local cũ khác hash để kiểm chứng thay thế.

## Kiểm tra source và phạm vi công cụ

AST ghi nhận {len(sites)} vị trí `strict` không hằng True; không phải tất cả đều là lỗi checkpoint trong đường báo cáo. {link(path,'DATASET/logs/phase7/task_7.16/strict_loading_review.json')} phân biệt initialization, diagnostic và legacy. Không xóa chuỗi grep hoặc đổi source loader để làm đẹp DoD. E1–E6 trong pipeline hiện hành được đối chiếu cụ thể ở phụ lục A.9 của kế hoạch CFE.

Kiểm thử mục tiêu hiện tại: **{tests['cases']} test pipeline + {tests['additional_components']['cases']} test thành phần = {tests['total_disjoint_cases']} passed, 0 failed/error/skipped**, trên 9 file riêng biệt; phạm vi ghi trong {link(path,'DATASET/logs/phase7/task_7.16/validation/targeted_tests.json')}. Không tuyên bố đã chạy xanh toàn bộ repository. Cross-review và chữ ký A/B/C vẫn chờ người thật.
'''
    return body


def gate_table(data):
    s, report, *_ = data
    return '\n'.join([
        '| Cổng | Kết quả hiện hành | Bằng chứng / phần còn thiếu |',
        '| :--- | :--- | :--- |',
        '| G7.0 | **PENDING** theo DoD toàn repo | Canonical core: 3 checkpoint nạp 0/0, Gen 1 bị từ chối; còn loader legacy và CI chẩn đoán n3/n8 chưa đủ. |',
        '| G7.1 | **FAIL** | Gen 2: Clean Fake@Real90 '+f(report['gates']['G7.1']['selected_clean_fake_at_real90'])+'% <55%; Q30 AUC '+f(report['gates']['G7.1']['selected_q30_auc'])+'% <55%. |',
        '| G7.2 | **FAIL — 6/8** | Clean Fake@Real90 '+f(s['benchmark']['cfe']['clean']['metrics']['fake_at_real90'])+'% <65%; worst-subset Fake ACC '+f(s['acceptance']['G72']['criteria']['Worst_Subset_Fake_ACC']['value'])+'% <45%. |',
        '| G7.3 | **PENDING** | 20 ảnh và overhead L4 '+f(s['latency']['batches']['1']['overhead_delta_ms'],6)+' ms/ảnh đạt; thiếu reference tốt nhất theo điều kiện. |',
    ])


def task_records(data):
    s,report,catalog,gen2,train,tests,sites=data
    return [
      ('7.1','C','[~]','Có audit 4 checkpoint; Gen 1 invalid_layout; n3/n8 verdict inconclusive vì thiếu CI phù hợp.'),
      ('7.2','B','[~]','Canonical core và từ chối layout đã xác minh; repo-wide legacy/exception review chưa đóng.'),
      ('7.3','C','[~]','Metric/CI đã đo; raw corrected schema v1 và view tài liệu v3 tách rõ; thiếu phân tầng generator chuẩn.'),
      ('7.4','C','[x]','Calibration runtime 400 ảnh đã hội tụ; T='+f(s['calibration']['T'],8)+'; FPR calibration 5%, không bảo đảm mọi miền.'),
      ('7.5','C','[x]','12 ứng viên measured; chọn α='+f(gen2['metadata']['alpha'])+', β='+f(gen2['metadata']['beta_srm'])+'; metadata nhúng và actual hash khớp matrix.'),
      ('7.6','C','[x]','Đã đo đủ 4 cấu hình ×2 điều kiện; kết quả G7.1 FAIL, không phải lỗi chưa chạy.'),
      ('7.7','A','[x]','14.400 bản ghi; train/val_calib/val_select=12.240/1.080/1.080; audit 0 vi phạm trong reference đã ghi.'),
      ('7.8','A','[x]','Pipeline v2 và benchmark CPU synthetic: overhead '+f(s['degradation']['pct_overhead'])+'% <15%; không suy thành latency CLIP/GPU.'),
      ('7.9','B','[x]','Loss v2 có code/smoke; kiểm thử thành phần được ghi riêng. Không đồng nghĩa đã chứng minh tăng khả năng tổng quát.'),
      ('7.10','B','[x]','Selection worst-group đã dùng trong training; CPU toy overhead '+f(s['selection']['ratio'],3)+'×, không phải CLIP benchmark.'),
      ('7.11','C','[~]','Có '+str(train['epochs'])+' epoch, chọn epoch '+str(train['selected_epoch'])+'; val_select worst '+f(train['selected_worst_group_pct'])+'%; log grad_norm NaN còn ở epoch '+','.join(map(str,train['nonfinite_grad_norm_epochs']))+'.'),
      ('7.12','C+B','[~]','Có ablation 4 cấu hình/seed42; chưa chứng minh contribution tổng quát của hai loss; không đổi gate từ report ablation.'),
      ('7.13','B','[~]','Router code có regression; experiment deferred, deployment tắt; số synthetic v1 thu hồi; fixed fallback dùng được.'),
      ('7.14','C','[~]','Đã chấm package thực, đủ dự đoán và CI; G7.2 FAIL 6/8; taxonomy/schema còn ghi rõ giới hạn.'),
      ('7.15','C','[~]','20 ảnh và overhead L4 đạt; G7.3 chờ reference; weights mới trên Drive, không trong ZIP local.'),
      ('7.16','A+B+C','[~]','Biên soạn/sync tài liệu và kiểm tra tự động hoàn tất; chờ cross-review và chữ ký thật A/B/C.'),
    ]


def task_table(data, selected=None):
    rows=['| Task | Chủ trì | Trạng thái | Kết quả và giới hạn |','| :--- | :--- | :--- | :--- |']
    rows += [f'| {key} | {owner} | {status} | {text} |' for key,owner,status,text in task_records(data) if selected is None or key in selected]
    return '\n'.join(rows)


def metric_table(data):
    s=data[0];rows=['| Điều kiện | AUC % [CI 95%] | AP % [CI 95%] | Fake@Real90 % [CI 95%] | ECE % |', '| :--- | :--- | :--- | :--- | ---: |']
    for name in ['clean','q30']:
        m=s['benchmark']['cfe'][name]['metrics'];ci=m['ci95']
        interval=lambda k:f(m[k])+' ['+f(ci[k][0])+'; '+f(ci[k][1])+']'
        rows.append(f"| {name} | {interval('auc')} | {interval('ap')} | {interval('fake_at_real90')} | {f(m['ece'])} |")
    return '\n'.join(rows)


def data_section(data):
    s,report,catalog,*_=data
    d=s['data'];counts=d['integrity']['summary'];download=catalog['data_download_bytes_reported']
    return f'''## Dữ liệu và chi phí đã có bằng chứng

Kho v2 có **{count(counts['n_records'])} bản ghi**: train **{count(counts['by_split']['train'])}**, val_calib **{count(counts['by_split']['val_calib'])}**, val_select **{count(counts['by_split']['val_select'])}**. 5 họ fake và real; {counts['n_generators']} generator/source labels, {counts['n_sources']} nguồn; replay {count(counts['replay'])} bản ghi. Đây không phải tập CFE calibration 400/evaluation 1.600 của package.

Ảnh được manifest ghi nhận **{count(counts['n_bytes_images'])} byte**; tổng `bytes_downloaded` của ba provenance là **{count(download)} byte**. Không cộng hai đại lượng này làm dung lượng ảnh. Audit có `violations=0`, `coverage.complete=true`, external pHash threshold8/internal3 trong các reference đã ghi; không diễn giải thành chứng minh tuyệt đối không có mọi dạng leakage. Audit dùng reference 18 tập cùng các tập bổ sung, không phải tuyên bố 82.003 nội dung ảnh độc lập.

Audit mất **{f(d['elapsed_s'],1)} giây**. **Chưa có số đo tổng thời gian stream**; không dùng thời gian audit làm thời gian tải. Không có bằng chứng đã kích hoạt fallback8.000 ảnh. `blocked=[]` trong audit nói không có generator vi phạm trong manifest cuối; không phải số generator bị bộ thu thập loại từ đầu.

Degradation v2 overhead **{f(s['degradation']['pct_overhead'])}%**, nguồn benchmark là32ảnh noise RGB, CPU/DataLoader, batch8, không phải tốc độ GPU hay quality benchmark. Selection overhead **{f(s['selection']['ratio'],3)}×**, mô hình toy linear CPU, không gán sang CLIP.
'''


def checkpoint_section(data):
    s,report,catalog,gen2,*_=data
    rows=['| Checkpoint | Tensor entries | Adapter blocks | Kết luận |','| :--- | ---: | :--- | :--- |']
    for item in s['structure']['checkpoints']:
        name=Path(item['path']).name
        rows.append(f"| `{name}` | {item['n_keys']} | `{item['adapter_blocks']}` | {'invalid_layout; chỉ forensic' if not item['expressible_n'] else 'layout biểu diễn được; xem canonical 0/0'} |")
    diag=s['gen1']['runs']
    return '\n'.join(rows)+f'''

### 5.4. Chẩn đoán Gen 1 bằng số thật

Gen 1 có10khối hợp của n3/n8, không có n hợp lệ biểu diễn đủ. Diagnostic partial trên2.000CFE, T1/ngưỡng0,5: n3 ACC **{f(diag[0]['acc'])}%**, Real **{f(diag[0]['real_acc'])}%**, Fake **{f(diag[0]['fake_acc'])}%**, AUC **{f(diag[0]['auc'])}%**; n8 ACC **{f(diag[1]['acc'])}%**, Real **{f(diag[1]['real_acc'])}%**, Fake **{f(diag[1]['fake_acc'])}%**, AUC **{f(diag[1]['auc'])}%**. Các số diagnostic này chỉ để tham khảo lịch sử, `invalid_layout`, không dùng làm hiệu năng của Gen 1 hợp lệ.

Delta Fake n8−n3 **{f(s['gen1']['delta_fake_acc'])} pp**, nhưng CI đối chiếu chưa có; giữ verdict **inconclusive**. Không suy ra different_basin, không chứng minh cần hai model và không cứu checkpoint bằng nạp thiếu.

### 7. Gen 2 đã đo, không nhầm metadata cũ

CSV có12ứng viên và đúng1selected: **α={f(gen2['metadata']['alpha'])}, β_srm={f(gen2['metadata']['beta_srm'])}**, n_keys={gen2['metadata']['n_keys']}, layout n3/[7,15,23]. Đã băm file local: **`{gen2['sha256']}`**, đọc metadata nhúng và đối chiếu matrix. Metadata rời `wise_ft_v2_best_metadata.json` α0,50 là ứng viên cũ `unselected`, hash khác; không dùng thay bản được chọn.

Gen 2 selected Clean AUC **{f(report['gates']['G7.1']['selected_clean_auc'])}%**, Fake@Real90 **{f(report['gates']['G7.1']['selected_clean_fake_at_real90'])}%**, Q30 AUC **{f(report['gates']['G7.1']['selected_q30_auc'])}%**: **G7.1 FAIL**. File nguồn matrix còn dùng key ngưỡng55% cho Fake@Real90 Q30, khác vị trí clean trong master plan; ghi nhận bất nhất này, không sửa raw. Theo tiêu chí clean55% và Q30AUC55% của kế hoạch, verdict vẫn FAIL.

Gen 2 là thí nghiệm trước, **không phải package đang triển khai**. Package hiện hành là Anchored v2 source`5cd49252…`, đóng gói hash`24869052…`.
'''


def defects_section():
    return '''## A.9. Trạng thái E1–E6 trong đường báo cáo hiện hành

| Mã | Hiện trạng | Giới hạn |
| :--- | :--- | :--- |
| E1 | Final checkpoint qua canonical loader, 0/0 hoặc từ chối có tên | Loader legacy còn trong repo; G7.0 rộng chưa đóng |
| E2 | Không ép Gen 1 về n3/n8; layout không đầy đủ bị từ chối | Partial chỉ là forensic, gate-ineligible |
| E3 | T/ngưỡng fit trên 400 calibration, tách 1.600 evaluation, không phụ thuộc batch đang chấm | Threshold biên có khác biệt CPU/CUDA rất nhỏ; benchmark giữ cùng backend |
| E4 | Tham số/số đo router v1 synthetic thu hồi; router tắt | Chưa đo held-out win; fallback không đồng nghĩa router thành công |
| E5 | Có AUC/AP/TPR1/5/10, ECE/NLL và CI95; nhiệt độ có kiểm tra hội tụ | Raw corrected schema v1; view v3 cho tài liệu là derived, không đổi schema nguồn |
| E6 | Gen 2 cùng n3, 12-grid α×β, verify và actual hash đã đối chiếu | Metadata rời cũ khác hash được giữ như lịch sử, không coi là selected |

Không suy tên generator từ filename để chế bảng5×4: nguồn hiện cung cấp per-subset, chưa đủ nhãn generator chuẩn cho taxonomy đó. Không fit lên evaluation; không dùng `mean_w` của tập để đặt ngưỡng. Mốc báo cáo hiện tại là post-hoc confirmation, không phải kiểm thử mù mới.
'''


def target_content(owner,data):
    s,report,catalog,gen2,train,tests,sites=data
    header=f'''## Kết quả Phase 7 hiện hành — 09/11/2026

Run `{report['run_id']}`, package `{report['package_sha256']}`. Số liệu hiện hành: {link(owner,RUN+'/analysis/README.md','hậu kiểm chi tiết')}, {link(owner,'DATASET/logs/phase7/task_7.16/final_phase7_report.json','view JSON v3 cho tài liệu')} và {link(owner,'docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md','bảng nguồn số cũ → mới')}. View v3 được tổng hợp từ raw corrected schema v1, **không phải một phép đo mới**.

{gate_table(data)}

Hoàn tất đo/biên soạn không có nghĩa đạt tất cả mục tiêu. Không dùng số Gen 1 như hiệu năng hợp lệ; không khẳng định router thắng, không đổi FAIL/pending thành PASS. Chấp nhận GPU CUDA đủ VRAM; latency chỉ áp dụng L4 đã ghi. Xác nhận chéo/chữ ký A/B/C còn chờ, không tạo chữ ký tự động.
'''
    if owner.endswith('/ke_hoach.md'):
        specific='## Tuần 7–8: trạng thái 16 task theo bằng chứng\n\n'+task_table(data)+'''

### Rủi ro đã thực sự xảy ra

- Gen 1 trộn layout10khối không biểu diễn được: cách ly và chuyển sang Gen 2 cùng n3; không nâng số khi G7.1 không đạt.
- Router v1 thiếu bằng chứng measured: hệ số/số synthetic bị thu hồi, bản hiện hành dùng fixed fallback.
- Bộ fit T từng dừng sớm: đã sửa hội tụ; benchmark GPU mới thống nhất FP16/CUDA/batch16.
- G7.2 còn thiếu hai tiêu chí, G7.3 thiếu best-reference; chưa có căn cứ gọi checkpoint Pareto tối ưu toàn bộ hoặc nghiệm thu cuối.
'''
    elif owner.endswith('ke_hoach_cac_cot_moc_va_diem_dung.md'):
        specific='''## Điểm dừng 6 và quyết định chung cuộc

Có thể bàn giao hồ sơ thí nghiệm và negative result có truy vết; chưa được tuyên bố toàn Phase 7 PASS. Chưa có Pareto/best-reference đủ bằng chứng thì không tự gọi checkpoint là tối ưu. Hoàn tất tài liệu không tự đóng G7.0/G7.3 hoặc thay chữ ký của hội đồng/ba thành viên.

'''+task_table(data,{'7.14','7.15','7.16'})
    elif owner.endswith('tong_hop_checkpoints_va_lich_su_thuc_nghiem.md'):
        specific=checkpoint_section(data)
    elif owner.endswith('bao_cao_kiem_dinh_communityforensics_eval.md'):
        specific='## IX. Số đo mới thay cho bảng A/B lịch sử\n\n'+metric_table(data)+f'''

Nguồn: {link(owner,RUN+'/raw/benchmark/final_report.json')} `/cfe`, {link(owner,RUN+'/analysis/ci_verification.json')}. Bảng A/B cũ dưới phần lịch sử không còn là kết quả hiện hành. Fake@Real90 là metric mốc Real90, khác Fake ACC tại ngưỡng deployment. Ngưỡng hiện hành **{f(s['calibration']['T_star_default'],8)}**, T **{f(s['calibration']['T'],8)}**.

In-dist macro balanced ACC **{f(s['benchmark']['indist']['macro_acc'])}%**, pooled Real ACC **{f(s['benchmark']['indist']['pooled_real_acc'])}%**. Không nhầm pooled accuracy93,46% thành macro. CI trên ảnh không khẳng định độc lập giữa mọi generator/dataset.

Clean Fake@Real90 thiếu **{f(report['gaps_pp']['CFE_clean_Fake@Real90'])} pp**, worst-subset Fake ACC thiếu **{f(report['gaps_pp']['Worst_Subset_Fake_ACC'])} pp**. FPR Q30 tại ngưỡng cố định **8,625%**, nên calibration5% không bảo đảm5% mọi miền. Không kết luận vượt baseline có ý nghĩa thống kê từ các protocol khác nhau.
'''
    elif owner.endswith('ke_hoach_test_communityforensics_eval.md'):
        specific=defects_section()+f'''

## Protocol đã đo

400 CFE calibration(200/200), 1.600 evaluation(800/800); 7 điều kiện, FP16/CUDA/batch16, CLIP resize cạnh ngắn224 bicubic/center crop, degradation post-resize. In-dist82.003lượt/18tập; chỉ báo cáo số lượt, không khẳng định82.003nội dung độc lập. Rawschema hiện hành `phase7_corrected_benchmark/v1`; view tài liệu `report_schema:v3` ghi rõ derived và nguồn hash.

{metric_table(data)}
'''
    elif owner.endswith('README_A.md'):
        specific=task_table(data,{'7.7','7.8'})+'\n\n'+data_section(data)
    elif owner.endswith('README_B.md'):
        specific=task_table(data,{'7.2','7.9','7.10','7.13'})+f'''

Kiểm thử mục tiêu: **{tests['cases']} passed**, phạm vi canonical/calibration/router/acceptance ghi trong JSON test; không gọi đó là test toàn repo hoặc thí nghiệm router held-out. Loss smoke dùng tiny RGB student/teacher, không phải quality benchmark CLIP. So sánh loss4run/seed42 chưa chứng minh cải thiện tổng quát; `primary_driver=null` trong ablation.

Source active core: `src/models/canonical_loader.py`, `src/models/unified_fatformer.py`. Các điểm non-strict legacy/initialization được phân loại trong {link(owner,'DATASET/logs/phase7/task_7.16/strict_loading_review.json')}; không được lấy loader legacy tạo số hiện hành.
'''
    elif owner.endswith('README_C.md'):
        specific=task_table(data,{'7.1','7.3','7.4','7.5','7.6','7.11','7.12','7.14','7.15','7.16'})+f'''

Benchmark/source checkpoint runtime hash`5cd49252…` đã ràng buộc với package`24869052…`, nhưng log training vẫn có `grad_norm=NaN` ở epoch{','.join(map(str,train['nonfinite_grad_norm_epochs']))}; không tự viết “gradient luôn hữu hạn”. Preflight L4 và benchmark training Blackwell là hai bản ghi khác phiên, không gộp thành một cấu hình GPU duy nhất. Không bắt buộc T4 cho run hiện tại.
'''
    elif owner.endswith('overall/README.md'):
        specific='## Tài liệu dùng để viết báo cáo và bàn giao\n\n'+'\n'.join('- '+link(owner,name) for name in OUTPUTS)+f'''

- {link(owner,'DATASET/logs/phase7/task_7.16/evidence_catalog.json','Catalog nguồn và JSON pointer')}
- {link(owner,RUN+'/README.md','Hồ sơ package thực đã hậu kiểm')}
- {link(owner,'docs/tasks/member_c_training_eval/phase_7/task_7.16_cap_nhat_tai_lieu_du_an.md','Đặc tả và checklist 7.16')}

Kế hoạch quy định mục tiêu; JSON measured quy định kết quả. Không dùng kế hoạch hoặc ảnh lịch sử làm số đo hiện hành. Các tài liệu trước đây bên dưới chỉ để truy vết.
'''
    elif owner.endswith('tong_quan_du_an.md'):
        specific=f'''## Kiến trúc đang được đo, không phải thiết kế router dự kiến

```mermaid
flowchart LR
  A[Ảnh] --> B[CLIP resize và center crop]
  B --> C[Anchored v2: SRM và frequency gating]
  C --> D[Logits: một backbone forward]
  D --> E[Softmax với T đã fit trên 400 calibration]
  E --> F[Ngưỡng cố định]
  F --> G[prob_fake và label]
```

Frequency gating λ(x) trong backbone khác DegradationRouter. Router hiện **tắt**, không tự chọn clean/JPEG ở deployment. Trọng số lấy từ Task7.11, không phải WiSE-FT Gen2. Package nhúng trọng số/calibration/metadata, bố trí n3/[7,15,23]; hash`{report['package_sha256']}`. Model mới ở Drive, local chỉ có log và metadata/reference, không thay bằng weights cũ.

Chất lượng đã đo FP16/CUDA/batch16; CLI ảnh mặc định FP32 chưa được gắn tự động với quality claim này. FPR cố định trên Q30 vượt5%; chưa bảo đảm báo oan5% mọi miền. G7.3 thiếu best-reference nên chưa chứng minh một model đạt mọi điều kiện tốt nhất, cũng không chứng minh bắt buộc phải có hai model.

{checkpoint_section(data)}
'''
    else:
        specific=f'''## Phạm vi hiệu lực của báo cáo Phase 6-R

Toàn bộ phần gốc bên dưới chỉ để tham khảo lịch sử; cột WiSE-FT Gen1 là `invalid_layout`. Thu hồi cách đọc “PASS100% mọi hard gate”, “96.58% macro” hoặc “hai chế độ là tất yếu”. Không so trực tiếp ACC lịch sử/partition cũ với AUC hoặc macro balanced hiện hành.

{metric_table(data)}

Số liệu trên thuộc run mới, không sửa artifact Phase6-R để giả là đã đo lại. Kết quả âm hiện hành và giới hạn ở {link(owner,'docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md')}. Bản Gen2 được đo riêng ở Task7.5–7.6, G7.1 FAIL, không cứu các số Gen1.
'''
    return header+'\n\n'+specific


def mark_old_numbers(text):
    pattern=re.compile(r'(?<!\d)(?:52[.,]05|4[.,]10|96[.,]58)(?!\d)')
    lines=[];fenced=False
    for line in text.splitlines():
        if line.strip().startswith('```'):fenced=not fenced
        if pattern.search(line) and not fenced and not any(word in line for word in ['invalid_layout','chỉ để tham khảo lịch sử']):
            if line.rstrip().endswith('|'):
                line=line.rstrip()[:-1]+' <br>*(chỉ để tham khảo lịch sử; không phải số hiện hành)* |'
            else:line+=' — *(chỉ để tham khảo lịch sử; không phải số hiện hành)*'
        lines.append(line)
    return '\n'.join(lines)


def render_target(owner,data):
    original=LOG/'original_documents'/owner
    if not original.exists():
        original.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/owner,original)
    original_text=original.read_text(encoding='utf-8')
    title,_,body=original_text.partition('\n')
    return title+'\n\n'+START+'\n'+readable(target_content(owner,data))+'\n'+END+'''

<details>
<summary>Kế hoạch nền và hồ sơ các giai đoạn trước — chỉ để tham khảo lịch sử, không dùng như kết quả Phase 7 hiện hành</summary>

Các bảng, hình, kết luận và lệnh dưới đây được giữ để truy vết. Số Gen 1 là `invalid_layout`; chỉ mục nguồn và nội dung hiệu lực nằm ở khối cập nhật phía trên. Không dùng kết luận “PASS toàn bộ” trong hồ sơ cũ thay cho phán quyết mới.

'''+mark_old_numbers(body)+'\n\n</details>\n'


def readable(text):
    """Spacing of prose units only; never rewrite hex hashes or numeric precision."""
    units='ảnh|lượt|khối|cặp|bản|tập|tensor|generator|subset|phút|giây|điều|warmup|ứng viên|ứng|nội dung|evaluation|calibration|record|cell|real|fake|pp|ms|GiB|GB|lần|mẫu|selected|CFE|AUC|ACC'
    text=re.sub(rf'(?<=\d)(?=(?:{units}))',' ',text)
    text=re.sub(r'(?<!\w)(có|hợp|Sweep|trên|sau|tách|khỏi|và|gồm|bao gồm|hơn|thiếu|đạt|chỉ|đúng|cho|mốc|ngưỡng|vượt|dưới|trong|khoảng|từ|nguồn|thể|đủ|val_calib|reference|fallback|batch|seed|epoch|bootstrap|Real ACC|CFE|FPR|Q30 AUC)(?=\d)',r'\1 ',text)
    text=re.sub(r'(?m)^-\s*(?=\d)','- ',text)
    for old,new in [('thực(','thực ('),('file`.pth`','file `.pth` '),('batch1/16','batch 1/16'),('calibration(','calibration ('),('evaluation(','evaluation ('),('ngắn224','ngắn 224'),('Rawschema','Raw schema'),('stretch224','stretch 224'),('báo oan5%','báo oan 5%'),('là32','là 32'),('loss4run','loss: 4 run')]:
        text=text.replace(old,new)
    for old,new in [('Task7.','Task 7.'),('Phase6-R','Phase 6-R'),('Phase7','Phase 7'),('batch16','batch 16'),('batch8','batch 8'),('layout10','layout 10'),('threshold8','threshold 8'),('internal3','internal 3'),('calibration5%','calibration 5%'),('bảo đảm5%','bảo đảm 5%'),('vượt5%','vượt 5%'),('source`','source `'),('hash`','hash `')]:
        text=text.replace(old,new)
    return text


def chapter(data):
    s,report,catalog,gen2,train,tests,sites=data
    owner='docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md'
    b=s['benchmark'];cal=s['calibration'];lat=s['latency']['batches']['1']
    return f'''# Chương 5 — Kết luận, đóng góp, hạn chế và hướng phát triển

Nguồn thực nghiệm: run `{report['run_id']}`, package `{report['package_sha256']}`. {link(owner,'DATASET/logs/phase7/task_7.16/final_phase7_report.json','View số liệu cho tài liệu')} được tổng hợp từ báo cáo GPU đã hậu kiểm; không phải một lần đo mới. Toàn bộ chương sử dụng metric có điều kiện vận hành, không thay AUC bằng accuracy hoặc pooled bằng macro. Evaluation đã được xem trước; kết quả là **post-hoc confirmation**, không phải kiểm thử mù mới.

## 5.1. Kết luận và trả lời câu hỏi nghiên cứu

### Câu hỏi 1: Độ tin cậy của kết luận có phụ thuộc vào cách nạp checkpoint và đo lường không?

Có. Audit cấu trúc cho thấy Gen 1 chứa hợp10khối adapter, không biểu diễn đầy đủ được bằng số khối hỗ trợ của constructor. Nạp partial n3 hoặc n8 tạo ra hai phép đo khác nhau, không phải cùng một model hợp lệ. Diagnostic n3/n8 chỉ để tham khảo lịch sử, `invalid_layout`; verdict hiện vẫn **inconclusive** vì không có CI đối chiếu phù hợp. Canonical loader xử lý đúng ba checkpoint hợp lệ và từ chối Gen 1 có tên lỗi. Đó là bằng chứng xác định về cấu trúc, không cần gắn CI thống kê cho số lượng khối. Tuy nhiên G7.0 rộng vẫn pending do các đường legacy và phạm vi kiểm tra chưa đóng toàn repo.

WiSE-FT Gen 2 sửa điều kiện tương thích bằng cách trộn Baseline và Anchored cùng n3. Sweep12ứng viên chọn α={f(gen2['metadata']['alpha'])}, β_srm={f(gen2['metadata']['beta_srm'])}, actual hash`{gen2['sha256']}`. Gen 2 không tạo quick win: Clean Fake@Real90 {f(report['gates']['G7.1']['selected_clean_fake_at_real90'])}% và Q30 AUC {f(report['gates']['G7.1']['selected_q30_auc'])}% không đạt mốc55%. Số Gen 1 cũ không được dùng làm bằng chứng ưu thế hoặc định lý rằng phải có hai model.

### Câu hỏi 2: Mô hình sau huấn luyện và hiệu chuẩn có đáp ứng mục tiêu độ bền không?

Mới đáp ứng một phần. Package Anchored v2 được chấm trực tiếp trên1.600CFE evaluation tách khỏi400calibration, với FP16/CUDA/batch16 và canonical CLIP preprocessing:

{metric_table(data)}

Nguồn: {link(owner,RUN+'/raw/benchmark/final_report.json')} `/cfe`; CI95 bootstrap1.000lần/seed42 đã được tính lại từ dự đoán. Clean Fake@Real90 {f(b['cfe']['clean']['metrics']['fake_at_real90'])}% có CI [{f(b['cfe']['clean']['metrics']['ci95']['fake_at_real90'][0])}; {f(b['cfe']['clean']['metrics']['ci95']['fake_at_real90'][1])}] bao gồm65%. Vì vậy chỉ kết luận điểm ước lượng không đạt cổng, không gọi đó là bằng chứng thống kê rằng năng lực tổng thể thấp hơn65%.

Trên18tập in-dist, macro là trung bình balanced accuracy của từng tập: **{f(b['indist']['macro_acc'])}%**, pooled Real ACC **{f(b['indist']['pooled_real_acc'])}%**; pooled accuracy {f(b['indist']['pooled']['accuracy'])}% là đại lượng khác. Hai mốc90% đạt theo điểm ước lượng, nhưng không chứng minh mọi tập đều được bảo vệ hoặc không có quên mô hình. G7.2 chỉ đạt **6/8**; thiếu Clean Fake@Real90 **{f(report['gaps_pp']['CFE_clean_Fake@Real90'])} pp** và worst-subset Fake ACC **{f(report['gaps_pp']['Worst_Subset_Fake_ACC'])} pp**.

Nhiệt độ đã hội tụ T={f(cal['T'],8)}, ngưỡng={f(cal['T_star_default'],8)}. NLL calibration sau fit {f(cal['nll_after'],4)}, ECE {f(cal['ece_after'])}%, nhưng chỉ là kết quả trên tập fit. FPR calibration5% không bảo đảm ngoài miền; tại ngưỡng cố định Q30 có FPR8,625%. Temperature scaling sửa confidence, không tự tăng thứ tự phân biệt/AUC về mặt toán học.

### Câu hỏi 3: Có thể vận hành bằng một artifact thay vì chọn model thủ công không?

Về cấu trúc và đường chạy đã đo, có thể đóng gói một checkpoint với calibration/metadata, một backbone forward và fixed fallback. Log GPU đối chiếu1.200tensor và20ảnh dữ liệu thực(10real+10fake), sai số xác suất tối đa0. Overhead median batch1 trên L4 là **{f(lat['overhead_delta_ms'],6)} ms/ảnh**, p95 **{f(lat['overhead_delta_p95_ms'],6)} ms/ảnh**, dưới ngưỡng5ms; tổng package **{f(lat['package_per_image_ms'])} ms/ảnh** không phải overhead. Nguồn {link(owner,RUN+'/raw/real_image_verification.json')} và {link(owner,RUN+'/raw/latency_phase7.json')}.

Nhưng chưa thể trả lời rằng artifact này đạt chất lượng tốt nhất trong mọi điều kiện: G7.3 vẫn **pending** phép so sánh với reference tốt nhất cùng partition/protocol trong1,5pp. DegradationRouter tắt vì thí nghiệm held-out chưa có; frequency gating trong backbone không đồng nghĩa router đã hoạt động. Không suy từ pending thành bắt buộc hai chế độ hay router đã thua.

## 5.2. Đóng góp được bằng chứng hỗ trợ

1. **Đo lường có truy vết:** chuyển trọng tâm sang AUC/AP/TPR@FPR, CI, source hashes và phân biệt ngưỡng vận hành với metric ROC. Lưu dự đoán từng ảnh cho7điều kiện và18tập, kiểm tra nhãn/ID/đếm, không tạo số liệu thay thế.
2. **Phát hiện và cô lập lỗi kiến trúc:** chứng minh layout Gen 1 không thể nạp đầy đủ; canonical core từ chối thay vì im lặng bỏ khối. Gen 2 cùng layout được đo riêng, chấp nhận G7.1 FAIL thay vì dùng số Gen 1 làm kết luận.
3. **Quy trình hiệu chuẩn và bàn giao có kiểm soát:** tách400/1.600, sửa hội tụ nhiệt độ, giữ cùng precision/batch/backend cho calibration và benchmark, kiểm chứng package và latency cùng hash. Code loss/selection và ablation được cung cấp như hạ tầng nghiên cứu; không khẳng định mỗi module đều tạo cải thiện tổng quát.

## 5.3. Hạn chế và phạm vi diễn giải

**Mục tiêu chưa đạt.** G7.2 thiếu hai tiêu chí: Clean Fake@Real90 thiếu{f(report['gaps_pp']['CFE_clean_Fake@Real90'])}pp, worst-subset Fake ACC thiếu{f(report['gaps_pp']['Worst_Subset_Fake_ACC'])}pp. Nhóm `5_other_autoregressive` chỉ đạt30,86% Fake ACC. Không hạ ngưỡng theo evaluation hoặc đổi định nghĩa gate để che kết quả này. Negative result có giá trị mô tả nhưng không tự trở thành nghiệm thu đầy đủ hoặc chứng minh checkpoint Pareto tối ưu.

**Quy mô và tính độc lập.** Tổng pool CFE2.000 được tách400calibration/1.600evaluation, chỉ5subset. CI rộng và không suy được cho generator ngoài phạm vi đã đo. Kho train v2 14.400 ảnh với val_calib1.080 là một nguồn khác, không được gọi nhầm là tập400CFE. In-dist82.003 là số lượt/bản ghi; có candidate dùng chung real reference giữa các tập, chưa có đủ ảnh gốc để chứng minh nội dung độc lập. CI theo ảnh chưa mô tả hết phụ thuộc giữa dataset/generator.

**Điểm gộp che chênh lệch miền.** Macro in-dist chỉ vượt90% khoảng0,10pp. Mean-subset Real ACC khoảng85,79%, trong khi pooled Real ACC90,41%;9/18tập Real ACC dưới90%. CI chẩn đoán resample theo18subset khoảng[86,76%;93,32%], không phải quy tắc gate mới. Không biến điểm vượt mốc thành bảo đảm ổn định giữa các miền.

**Giới hạn ablation.** Bốn cấu hình cùng seed42; report không có CI độc lập cho selection worst-group và `primary_driver=null`. Không thể tách riêng đóng góp dữ liệu khi cả bốn đều dùng v2; không thể tuyên bố RealGuard/FeatureConsistency chắc chắn nâng chất lượng ngoài miền. Log training còn `grad_norm=NaN` ở epoch{','.join(map(str,train['nonfinite_grad_norm_epochs']))}; điều này chưa được giải thích, nhưng cũng không tự chứng minh weights cuối chứa NaN vì các dự đoán hiện hành hữu hạn.

**Protocol và khả năng triển khai.** In-dist cũ stretch224×224 khác preprocessing mới, nên so sánh trước/sau không quy hết cho calibration. Chất lượng hiện tại là FP16/CUDA/batch16; CLI ảnh thông thường mặc định FP32. Tự tính lại CPU softmax có1mẫuQ50 và5mẫuin-dist đổi nhãn sát ngưỡng dù sai khác xác suất≤3×10⁻⁷. Không chuyển quality claim sang mọi backend/batch mà chưa xác minh.

**Giới hạn bàn giao và quản trị.** Model mới ở Drive; ZIP không kèm weights nên hậu kiểm local không băm lại file`.pth`24869052… hoặc source checkpoint. G7.3 thiếu best-reference; G7.0 còn legacy. Raw benchmark dùng schema corrected/v1; view v3 trong hồ sơ này chỉ là bản tổng hợp, không thay schema gốc. Không có taxonomy generator chuẩn để tự tạo bảng per-generator. Chữ ký và cross-review A/B/C chưa có, không được giả lập.

## 5.4. Hướng phát triển và điều kiện bàn giao tiếp

1. **Thí nghiệm chất lượng có kiểm soát:** audit nhóm khó và nhãn nguồn, chọn thay đổi trên train/val_select; nhiều seed, paired CI và tập test mới chưa xem. Đối chiếu reference cùng partition/precision/preprocessing trước khi chốt G7.3. Không tái dùng evaluation để chọn T/ngưỡng.
2. **Độ tin cậy vận hành:** đồng bộ CLI với protocol đã đánh giá, kiểm tra batch1/16 và CPU/CUDA quanh biên ngưỡng; đo I/O và preprocessing nếu cần latency đầu-cuối. Đánh giá FPR theo miền và calibration chuyên điều kiện mà không làm rò rỉ test.
3. **Tái lập và nghiệm thu:** đưa đúng weights cùng hash vào bàn giao, xử lý hoặc phân loại loader legacy, giải thích gradient telemetry và reconcile metadata; thu thập cross-review/chữ ký thật. Router chỉ được bật sau fit/evaluation held-out hợp lệ, không từ số synthetic hoặc unit test.

Hồ sơ này cho phép báo cáo một kết quả thực nghiệm có kiểm chứng và các mục chưa đạt, không tuyên bố toàn bộ Phase 7 đã PASS. Các nguồn chi tiết, số cũ→mới và kiểm tra tự động nằm ở {link(owner,'docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md')}.
'''


def slides(data):
    s,report,catalog,gen2,train,tests,sites=data
    owner='docs/bao_cao/overall/slide_phase7_5phut.md';b=s['benchmark'];lat=s['latency']['batches']['1']
    source=link(owner,'DATASET/logs/phase7/task_7.16/final_phase7_report.json','JSON tổng hợp có nguồn/hash')
    pages=[]
    pages.append(f'''# Slide 1 — Vì sao phải kiểm tra lại cách đo? (45 giây)

Gen 1 có hợp10khối không biểu diễn được; không lấy ACC của partial load làm chất lượng model. Số Gen 1 cũ là `invalid_layout`, chỉ để tham khảo lịch sử.

```mermaid
flowchart LR
  A[Baseline n3: 7,15,23] --> C[Gen1: hợp 10 khối]
  B[M4 n8: 2,5,8,11,14,17,20,23] --> C
  C --> D[Không có n nạp đầy đủ]
  D --> E[Canonical loader từ chối]
```

Nguồn: {link(owner,'DATASET/logs/phase7/wise_ft_sweep/checkpoint_structure_audit.json')} `/checkpoints`. Hình là minh họa cấu trúc, không phải số đo chất lượng/CI.
''')
    pages.append(f'''# Slide 2 — Cách sửa và protocol (50 giây)

- Canonical final load: nạp đủ hoặc từ chối có tên; không im lặng ép layout.
-400CFE calibration tách1.600evaluation; train v2/val_calib1.080 là nguồn khác.
- NLL có kiểm tra hội tụ; calibration và chấm dùng FP16/CUDA/batch16. Latency đo riêng batch1/8/16/32.
- Lưu logits, xác suất, nhãn, ID và hash; CI95 bootstrap1.000lần/seed42.
- Router tắt, dùng fixed fallback; không có chiến thắng router held-out.

Nguồn: {source}, `/calibration`, `/environment`, `/source_reports`.
''')
    rows=['| Cấu hình 7.6 | Clean AUC % [CI95] | Clean Fake@Real90 % [CI95] | Q30 AUC % [CI95] |','| :--- | :--- | :--- | :--- |']
    for model,conditions in s['matrix']['results'].items():
        x=conditions['clean'];y=conditions['q30']
        ci=lambda m,k:f(m[k])+' ['+f(m['ci95'][k][0])+'; '+f(m['ci95'][k][1])+']'
        rows.append(f"| {model} | {ci(x,'auc')} | {ci(x,'fake_at_real90')} | {ci(y,'auc')} |")
    pages.append('''# Slide 3 — G7.0 và G7.1: sửa phép đo không tự bảo đảm quick win (50 giây)

G7.0: canonical core có bằng chứng, DoD legacy toàn repo còn pending. G7.1: **FAIL**, không có cấu hình đạt các mốc clean/Q30 đã định.

'''+'\n'.join(rows)+f'''

Gen 2 selected α={f(gen2['metadata']['alpha'])}, β={f(gen2['metadata']['beta_srm'])}; actual hash khớp metadata nhúng và matrix. Nguồn: {link(owner,'DATASET/logs/phase7/wise_ft_sweep/cfe_v3_baseline_matrix.json')} `/results`; lưu ý key ngưỡng raw cũ được giải thích trong hồ sơ, không đổi raw.
''')
    rows=['| Tiêu chí G7.2 | Điểm % | Mốc % | CI95 nếu có trong nguồn | Kết quả |','| :--- | ---: | :--- | :--- | :--- |']
    for key,item in s['acceptance']['G72']['criteria'].items():
        value=item['value']*100 if key=='ECE_Post_Calibration' else item['value']
        ci='Chưa có CI riêng trong gate'
        if key.startswith('CFE_'):
            cond='clean' if '_clean_' in key else 'q30';field='auc' if key.endswith('AUC') else 'fake_at_real90'
            bounds=b['cfe'][cond]['metrics']['ci95'][field];ci=f'[{f(bounds[0])}; {f(bounds[1])}]'
        target='<=8,00' if key=='ECE_Post_Calibration' else item['target']
        rows.append(f"| {key} | {f(value)} | {target} | {ci} | {'PASS' if item['pass'] else '**FAIL**'} |")
    pages.append('''# Slide 4 — G7.2: 6/8, tổng thể FAIL (65 giây)

'''+'\n'.join(rows)+f'''

Thiếu Clean Fake@Real90 **{f(report['gaps_pp']['CFE_clean_Fake@Real90'])} pp** và worst-subset Fake ACC **{f(report['gaps_pp']['Worst_Subset_Fake_ACC'])} pp**. Không đổi gate theo kết quả. Nguồn: {source} `/gates/G7.2`, `/cfe`.
''')
    pages.append(f'''# Slide 5 — Một package đã chạy; G7.3 vẫn pending (55 giây)

- Hash package `24869052…`; log đối chiếu1.200tensor và20ảnh lấy từ calibration (10 real + 10 fake), sai số xác suất0. Đây là kiểm tra nhất quán đóng gói, không phải tập test chất lượng độc lập mới.
- L4/FP16: overhead batch1 **{f(lat['overhead_delta_ms'],6)} ms/ảnh**, p95 **{f(lat['overhead_delta_p95_ms'],6)} ms/ảnh**; tổng package **{f(lat['package_per_image_ms'])} ms/ảnh**.
-200cặp đo sau20warmup, không gồm I/O/preprocessing; CI latency không được báo, p95 không phải CI.
- Chưa có best-reference cùng điều kiện trong1,5pp; do đó **không chốt G7.3 PASS**.

Nguồn: {source} `/latency/batches/1`, `/real_image_verification`, `/gates/G7.3`. Weights ở Drive, local không có file mới để băm lại.
''')
    pages.append(f'''# Slide 6 — Hạn chế và việc tiếp theo (35 giây)

- CFE chỉ1.600evaluation/5subset; post-hoc, không phải blind test. CI theo ảnh không bao hết phụ thuộc generator/dataset.
- FPR Q30 **8,625%** dù calibration5%; in-dist macro chỉ vượt90% khoảng0,10pp, mean-subset Real thấp hơn pooled.
- Một seed ablation; chưa chứng minh contribution loss hoặc router thắng; gradient telemetry còn NaN.
- Bổ sung reference phù hợp, thí nghiệm nhiều seed/test mới; đóng legacy và thu cross-review/chữ ký thật.

Kết luận: có kết quả thực nghiệm truy vết được, **không tuyên bố toàn Phase7 PASS**. Nguồn: {link(owner,'docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md')} §5.3–5.4; {source}.
''')
    assert len(pages)==6
    return '\n---\n\n'.join(readable(page) for page in pages)


def milestone(data):
    s,report,catalog,gen2,train,tests,sites=data
    owner='docs/milestones/milestone_phase7_acceptance.md'
    return f'''# Biên bản trạng thái nghiệm thu Phase 7

Ngày lập hồ sơ:09/11/2026. Người lập bản tổng hợp: Codex, từ dữ liệu đã hậu kiểm. **Đây là biên bản kỹ thuật chờ con người xác nhận, không phải chữ ký hoặc quyết định nghiệm thu của A/B/C.**

## 1. Phạm vi và kết luận

Run `{report['run_id']}`, package `{report['package_sha256']}`. {link(owner,RUN+'/raw/acceptance.json','Phán quyết gốc')} và {link(owner,'DATASET/logs/phase7/task_7.16/final_phase7_report.json','view tổng hợp có nguồn')}. Model ở Drive, ZIP không có weights; local không băm lại actual package. Không lấy thí nghiệm/đo/tài liệu hoàn tất thay cho PASS tất cả gate.

{gate_table(data)}

G7.2 negative result có thể bàn giao để đánh giá khoa học với hạn chế rõ; chưa có chứng minh Pareto/best-reference và chữ ký, nên **không tuyên bố nghiệm thu toàn Phase7**.

## 2. Trạng thái 16 task

{task_table(data)}

## 3. Checklist 11 tài liệu bắt buộc

'''+ '\n'.join(f'- [x] {link(owner,p)} — đã cập nhật số thật, nguồn và phạm vi hiệu lực.' for p in TARGETS)+f'''

## 4. Hồ sơ bàn giao và kiểm tra máy

- [x] {link(owner,'docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md')} — số cũ→mới→JSON pointer.
- [x] {link(owner,'docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md')} — đủ5.1–5.4, số pp thiếu và CI/giới hạn.
- [x] {link(owner,'docs/bao_cao/overall/slide_phase7_5phut.md')} —6slide/300giây, số sinh từ cùng JSON.
- [x] {link(owner,'DATASET/logs/phase7/task_7.16/evidence_catalog.json')} và source hash; snapshot các tài liệu trước sửa trong `original_documents/`.
- [x]103test mục tiêu và53test thành phần pass: **{tests['total_disjoint_cases']} case trên9file riêng biệt**, không tuyên bố toàn repository pytest xanh.
- [x] Phân biệt current report path với các vị trí loader legacy/khởi tạo/diagnostic; không xóa grep để làm giả repo-wide sạch.
- [ ] Dọn hoặc phê duyệt rõ phạm vi legacy của G7.0; hiện còn pending, không thuộc việc ký khống để đóng7.16.
- [ ] Best-reference theo điều kiện cho G7.3; không tạo phép đo hoặc CI thay thế.

## 5. Cross-review và chữ ký thật

| Vai trò | Họ tên | Nội dung phải đọc chéo | Xác nhận/ngày ký |
| :--- | :--- | :--- | :--- |
| A | Chưa cung cấp | Kiến trúc/đóng gói của B+C; kiểm tra counts và phạm vi reference | **Chưa ký** |
| B | Chưa cung cấp | Báo cáo CFE/Chương5 của C và hồ sơ dữ liệu của A | **Chưa ký** |
| C | Chưa cung cấp | README_A/README_B, đối chiếu nguồn, metric, gate và slide | **Chưa ký** |

Không ai được đánh dấu đã đọc hoặc đã ký khi chưa có xác nhận thực. Chữ ký/đánh giá này không được tạo bởi script. Trạng thái7.16: **biên soạn và kiểm tra tự động hoàn tất; nghiệm thu chính thức chờ cross-review A/B/C**. Dòng này không thay trạng thái FAIL/pending của các gate.
'''


def backup_document(owner):
    destination=LOG/'original_documents'/owner
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/owner,destination)
    return destination.read_text(encoding='utf-8')


def supplementary_documents(data):
    """Update navigation and label retained historical numbers without rewriting evidence."""
    main_plan='docs/plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md'
    index='docs/tasks/member_c_training_eval/phase_7/README.md'
    spec='docs/tasks/member_c_training_eval/phase_7/task_7.16_cap_nhat_tai_lieu_du_an.md'
    outputs={}
    for owner in [main_plan,index,spec,'README.md']:
        original=backup_document(owner)
        if owner=='README.md':
            original=re.sub(r'### Lộ Trình Hiện Hành: Phase 7[^\n]*\n.*?(?=\n---)',
                '### Lộ trình Phase 7: trạng thái đo thực tế\n\n'+gate_table(data)+'\n\nTask 7.16 đã biên soạn tài liệu; cross-review/chữ ký A/B/C chưa có. Xem [biên bản nghiệm thu](docs/milestones/milestone_phase7_acceptance.md).\n',original,flags=re.S)
        if owner==spec:
            # Human confirmation and broad loading cleanup are deliberately not checked.
            for prefix in ['**11/11**','`bao_cao_ra_soat_so_lieu_phase7.md`','`grep "52.05','Chương 5','Slide 6','Không còn câu nào']:
                original=original.replace('- [ ] '+prefix,'- [x] '+prefix)
        if owner==index:
            original=re.sub(r'(\| \*\*7\.16\*\*[^\n]*?)`\[ \] Sẵn sàng`',r'\1`[~] Tài liệu xong; chờ cross-review A/B/C`',original)
        title,_,body=original.partition('\n')
        outputs[owner]=title+'\n\n'+START+f'''
## Trạng thái thực tế ngày 09/11/2026

Task 7.16 đã biên soạn 11/11 tài liệu, Chương 5, slide 6 trang và hồ sơ nghiệm thu. **Chờ cross-review và chữ ký thật A/B/C**; không xác nhận thay con người.

{gate_table(data)}

Hồ sơ: {link(owner,OUTPUTS[3],'biên bản và checklist')} và {link(owner,OUTPUTS[0],'bảng số cũ → mới → nguồn')}. GPU có thể là bất kỳ CUDA GPU đủ VRAM; kết quả latency hiện có chỉ được đo trên L4/FP16. Các mốc và số liệu cũ phía dưới là kế hoạch/hồ sơ lịch sử, không thay phán quyết hiện hành.
'''+END+'\n\n'+mark_old_numbers(body)+'\n'
    old_pattern=re.compile(r'(?<!\d)(?:52[.,]05|4[.,]10|96[.,]58)(?!\d)')
    for path in sorted((ROOT/'docs').rglob('*.md')):
        owner=path.relative_to(ROOT).as_posix()
        if owner in TARGETS+OUTPUTS or owner in outputs or 'sources' in path.parts:
            continue
        original=path.read_text(encoding='utf-8')
        if not old_pattern.search(original):continue
        original=backup_document(owner)
        title,_,body=original.partition('\n')
        outputs[owner]=title+'\n\n'+START+f'''
> Hồ sơ lịch sử / `invalid_layout` đối với số Gen 1. Các số 52.05, 4.10 và 96.58 dưới đây chỉ để tham chiếu lịch sử; không phải kết quả Phase 7 hiện hành. Xem {link(owner,OUTPUTS[0],'đính chính có nguồn')} và {link(owner,OUTPUTS[3],'trạng thái nghiệm thu ngày 09/11/2026')}. Không sử dụng kết luận hiệu lực cũ thay cho G7.1 FAIL, G7.2 FAIL và G7.3 pending.
'''+END+'\n\n'+mark_old_numbers(body)+'\n'
    return outputs


def legacy_number_ledger():
    result=[]
    pattern=re.compile(r'(?<!\d)(?:52[.,]05|4[.,]10|96[.,]58)(?!\d)')
    for path in sorted((ROOT/'docs').rglob('*.md')):
        if 'sources' in path.parts:continue
        text=path.read_text(encoding='utf-8')
        for number,line in enumerate(text.splitlines(),1):
            if pattern.search(line):
                result.append(dict(file=path.relative_to(ROOT).as_posix(),line=number,text=line,
                    classification='retained historical/invalid_layout, mapping or literal audit requirement',
                    scope_labeled=any(word in text for word in ['invalid_layout','chỉ để tham chiếu lịch sử','số cũ → mới'])))
    return result


def write_documents(data):
    assert (ROOT/OUTPUTS[0]).exists(), 'Run --catalog-only before writing target documents'
    documents={owner:render_target(owner,data) for owner in TARGETS}
    documents.update({OUTPUTS[0]:source_map_report(data),OUTPUTS[1]:readable(chapter(data)),
                      OUTPUTS[2]:slides(data),OUTPUTS[3]:readable(milestone(data))})
    documents.update(supplementary_documents(data))
    for owner,content in documents.items():
        path=ROOT/owner;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content,encoding='utf-8')
    # Never create or infer a human signature from completion of automatic checks.
    signoff_path=LOG/'human_signoff.json'
    if not signoff_path.exists():
        write_json(signoff_path,dict(status='pending_actual_cross_review',
            participants=[dict(role=role,name=None,signed=False,date=None) for role in ['A','B','C']],
            notice='Only actual human confirmation can close these fields; scripts must not sign.'))
    ledger=legacy_number_ledger()
    write_json(LOG/'legacy_number_review.json',dict(scope='authored docs/**/*.md excluding docs/sources',hits=ledger,
        unresolved_unlabeled=[item for item in ledger if not item['scope_labeled']]))
    owner='DATASET/logs/phase7/task_7.16/README.md'
    (LOG/'README.md').write_text(f'''# Task 7.16 — hồ sơ tài liệu Phase 7

**Biên soạn và kiểm tra tự động hoàn tất; chờ cross-review/chữ ký thật A/B/C.** Không xác nhận toàn Phase 7 PASS.

{gate_table(data)}

Hồ sơ để đọc theo thứ tự:

1. {link(owner,OUTPUTS[0],'Đối chiếu số cũ → mới → JSON pointer')}.
2. {link(owner,OUTPUTS[1],'Chương 5: kết luận và hạn chế')}.
3. {link(owner,OUTPUTS[2],'Slide 6 trang / 5 phút')}.
4. {link(owner,OUTPUTS[3],'Biên bản, 16 task và 11 tài liệu')}.

`evidence_catalog.json` / `source_values.csv`: số và hash nguồn. `final_phase7_report.json`: view tài liệu v3, **không phải raw GPU report mới**. `gen2_verified_metadata.json`: hash actual Gen 2 và xung đột sidecar cũ. `training_summary.json`: có NaN gradient telemetry. `strict_loading_review.json`: các ngoại lệ/legacy còn mở. `legacy_number_review.json`: mọi số cũ còn giữ kèm phạm vi lịch sử. `original_documents/`: snapshot trước sửa. `human_signoff.json`: chưa có chữ ký.

Kiểm tra tái lập từ root repo:

```powershell
python -m tools.sync_phase7_documentation --check
```

Kiểm thử: 103 case pipeline + 53 case loss/selection/degradation = **156 case trên 9 file**, không phải toàn bộ repository. XML và kiểm tra tài liệu ở `validation/`. Không sửa notebook, weights hoặc kết quả đo; model mới chưa được gửi về local nên không chạy lại forward model tại đây.
''',encoding='utf-8')
    protected={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'notebooks').rglob('*.ipynb')}
    imported=read(RUN+'/import_manifest.json')
    for item in imported['raw_files']:
        name=RUN+'/raw/'+item['path'];assert digest(ROOT/name)==item['sha256']
        protected[name]=item['sha256']
    authored=list(documents)+[owner]
    evidence=['evidence_catalog.json','final_phase7_report.json','source_values.csv','gen2_verified_metadata.json',
              'training_summary.json','strict_loading_review.json','legacy_number_review.json']
    manifested=authored+[str((LOG/name).relative_to(ROOT).as_posix()) for name in evidence]
    manifested+=['tools/sync_phase7_documentation.py']
    write_json(LOG/'documentation_manifest.json',dict(schema='phase7_documentation_manifest/v1',
        authored=authored,hashes={name:digest(ROOT/name) for name in manifested},protected=protected,
        formal_acceptance='pending_actual_cross_review_and_signoff',raw_files_checked=len(imported['raw_files'])))
    check_documents(data)


def current_text(path):
    text=path.read_text(encoding='utf-8')
    return text.split(START,1)[1].split(END,1)[0] if START in text else text


def check_documents(data):
    from urllib.parse import unquote
    manifest=read('DATASET/logs/phase7/task_7.16/documentation_manifest.json')
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
    for name,expected in {**manifest['hashes'],**manifest['protected']}.items():
        check('sha256:'+name,(ROOT/name).exists() and digest(ROOT/name)==expected)
    for fact in data[2]['facts']:
        check('source_pointer:'+fact['id'],pointer(read(fact['file']),fact['pointer'])==fact['value'])
    for source in data[2]['sources'].values():
        check('source_hash:'+source['path'],digest(ROOT/source['path'])==source['sha256'])
    for owner in TARGETS:
        check('generated_block:'+owner,current_text(ROOT/owner).strip()==readable(target_content(owner,data)).strip())
    for owner,expected in [(OUTPUTS[0],source_map_report(data)),(OUTPUTS[1],readable(chapter(data))),
                           (OUTPUTS[2],slides(data)),(OUTPUTS[3],readable(milestone(data)))]:
        check('generated_document:'+owner,(ROOT/owner).read_text(encoding='utf-8')==expected)
    slide_text=(ROOT/OUTPUTS[2]).read_text(encoding='utf-8')
    check('six_slides',re.findall(r'^# Slide (\d)',slide_text,re.M)==list('123456'))
    chapter_text=(ROOT/OUTPUTS[1]).read_text(encoding='utf-8')
    check('four_chapter_sections',all(re.search(r'^## 5\.'+str(i)+r'\b',chapter_text,re.M) for i in range(1,5)))
    missing=[]
    for owner in manifest['authored']:
        text=current_text(ROOT/owner)
        check('utf8_no_replacement:'+owner,'\ufffd' not in text)
        for destination in re.findall(r'\]\(([^)]+)\)',text):
            destination=destination.strip('<>').split('#',1)[0]
            if not destination or re.match(r'^[a-zA-Z]+:',destination):continue
            path=(ROOT/owner).parent/unquote(destination)
            if not path.exists():missing.append(dict(owner=owner,target=destination))
    check('current_relative_links_resolve',not missing)
    ledger=legacy_number_ledger()
    check('legacy_numbers_scope_labeled',all(item['scope_labeled'] for item in ledger))
    signoff=read('DATASET/logs/phase7/task_7.16/human_signoff.json')
    check('three_human_roles_recorded',sorted(p['role'] for p in signoff['participants'])==['A','B','C'])
    # These are evidence status fields, not a machine verdict that people have reviewed.
    check('no_automatic_human_signature',all(not p['signed'] and p['name'] is None and p['date'] is None for p in signoff['participants']))
    check('measured_gate_status_preserved',data[1]['gates']['G7.2']['pass'] is False and data[1]['gates']['G7.3']['status']=='pending')
    imported=read(RUN+'/import_manifest.json')
    check('executed_notebook_matches_submission',digest(ROOT/imported['notebook']['path'])==imported['notebook']['sha256'])
    result=dict(schema='phase7_documentation_check/v1',passed=all(c['passed'] for c in checks),
        total_checks=len(checks),checks=checks,missing_links=missing,
        documentation_status='complete_machine_checked' if all(c['passed'] for c in checks) else 'requires_fix',
        human_acceptance='pending_actual_cross_review_and_signoff',
        targeted_test_cases=data[5]['total_disjoint_cases'],raw_files_preserved=manifest['raw_files_checked'])
    write_json(LOG/'validation/documentation_check.json',result)
    failed=[item for item in checks if not item['passed']]
    print(json.dumps({key:result[key] for key in ['passed','total_checks','targeted_test_cases','raw_files_preserved','human_acceptance']},ensure_ascii=False))
    if failed or missing:
        print(json.dumps(dict(failed=failed,missing_links=missing),ensure_ascii=False,indent=2))
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-only',action='store_true')
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    data=build_catalog()
    report=ROOT/'docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md'
    if args.catalog_only:
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(source_map_report(data),encoding='utf-8')
        print('Evidence catalog and old-to-new source map created before modifying 11 targets.')
    elif args.write:
        write_documents(data)
    elif args.check:
        check_documents(data)
    else:
        parser.error('Choose --catalog-only, --write or --check')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
