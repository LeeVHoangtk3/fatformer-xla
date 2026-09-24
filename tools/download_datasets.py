# -*- coding: utf-8 -*-
"""
Script tu dong tai va giai nen cac bo du lieu cho FatFormer.
Ho tro resumable download (HTTP Range), theo doi tien do, logging,
va tu dong giai nen vao dung cau truc thu muc DATASET.
"""

import os
import sys

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import argparse
import zipfile
import requests

DATASET_CATALOG = {
    'val': {
        'name': 'ProGAN Validation Set (sywang/CNNDetection - 792 MB)',
        'files': [
            {
                'filename': 'progan_val.zip',
                'url': 'https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip',
                'expected_size': 830792545,
                'extract_to': 'val'
            }
        ]
    },
    'test_progan': {
        'name': 'ProGAN Test Set (sywang/CNNDetection - 795 MB)',
        'files': [
            {
                'filename': 'progan_testset.zip',
                'url': 'https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_testset.zip',
                'expected_size': 834177578,
                'extract_to': 'test'
            }
        ]
    },
    'test_gans': {
        'name': 'CNN Synth 8-family GANs Test Set (Wang et al. - 18.68 GB)',
        'files': [
            {
                'filename': 'CNN_synth_testset.zip',
                'url': 'https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip',
                'expected_size': 20052866587,
                'extract_to': 'test'
            }
        ]
    },
    'train': {
        'name': 'ProGAN Train 7z Multi-part (Wang et al. - 70 GB total)',
        'files': [
            {
                'filename': f'progan_train.7z.00{i}',
                'url': f'https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.00{i}',
                'expected_size': 10737418240 if i < 7 else 10499493436,
                'extract_to': None
            } for i in range(1, 8)
        ]
    }
}


def format_bytes(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def log_msg(msg, log_file=None):
    t_str = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{t_str}] {msg}"
    print(formatted, flush=True)
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + "\n")
        except Exception:
            pass


def download_file(url, target_path, expected_size=None, log_file=None):
    part_path = target_path + '.part'
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    downloaded = 0
    if os.path.exists(part_path):
        downloaded = os.path.getsize(part_path)

    if os.path.exists(target_path):
        current_size = os.path.getsize(target_path)
        if expected_size and current_size == expected_size:
            log_msg(f"File da ton tai va khop dung luong: {target_path} ({format_bytes(current_size)})", log_file)
            return True
        elif not expected_size and current_size > 0:
            log_msg(f"File da ton tai: {target_path} ({format_bytes(current_size)})", log_file)
            return True

    headers = {}
    if downloaded > 0:
        headers['Range'] = f"bytes={downloaded}-"
        log_msg(f"Tiep tuc tai noi (Resume) tu byte {downloaded} ({format_bytes(downloaded)})...", log_file)
    else:
        log_msg(f"Bat dau tai: {url}", log_file)

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    log_msg(f"Loi may chu HTTP {r.status_code}. Thu lai lan {attempt}/{max_retries}...", log_file)
                    time.sleep(3)
                    continue

                total_size = expected_size
                if 'content-length' in r.headers:
                    total_size = int(r.headers['content-length']) + downloaded

                mode = 'ab' if downloaded > 0 else 'wb'
                chunk_size = 1024 * 1024  # 1 MB chunks
                last_time = time.time()
                last_downloaded = downloaded

                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            now = time.time()
                            if now - last_time >= 5.0:
                                speed = (downloaded - last_downloaded) / (now - last_time)
                                pct = (downloaded / total_size * 100) if total_size else 0
                                eta_str = 'N/A'
                                if speed > 0 and total_size:
                                    eta_sec = (total_size - downloaded) / speed
                                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                                log_msg(
                                    f"Tien do: {format_bytes(downloaded)}/{format_bytes(total_size) if total_size else '?'} "
                                    f"({pct:.1f}%) | Toc do: {format_bytes(speed)}/s | ETA: {eta_str}",
                                    log_file
                                )
                                last_time = now
                                last_downloaded = downloaded

            if expected_size and downloaded < expected_size:
                log_msg(f"Canh bao: Chua tai du ({downloaded}/{expected_size}). Dang ket noi lai...", log_file)
                headers['Range'] = f"bytes={downloaded}-"
                continue

            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(part_path, target_path)
            log_msg(f"Da tai hoan tat: {target_path} ({format_bytes(downloaded)})", log_file)
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            log_msg(f"Loi ket noi ({e}). Thu lai {attempt}/{max_retries} sau 5s...", log_file)
            time.sleep(5)
            if os.path.exists(part_path):
                downloaded = os.path.getsize(part_path)
                headers['Range'] = f"bytes={downloaded}-"

    log_msg(f"That bai khi tai {url} sau {max_retries} lan thu.", log_file)
    return False


def extract_zip(zip_path, extract_dir, log_file=None):
    log_msg(f"Bat dau giai nen {zip_path} -> {extract_dir}...", log_file)
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        log_msg(f"Giai nen thanh cong vao: {extract_dir}", log_file)
        return True
    except Exception as e:
        log_msg(f"Loi giai nen {zip_path}: {e}", log_file)
        return False


def main():
    parser = argparse.ArgumentParser(description='FatFormer Dataset Downloader')
    parser.add_argument('--target', type=str, default='val',
                        choices=['val', 'test_progan', 'test_gans', 'train', 'all'],
                        help='Dataset target to download (val, test_progan, test_gans, train, all)')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root path of DATASET folder')
    parser.add_argument('--no_extract', action='store_true',
                        help='Download only without extraction')
    args = parser.parse_args()

    if args.dataset_root:
        dataset_root = os.path.abspath(args.dataset_root)
    else:
        cur = os.path.abspath(os.path.dirname(__file__))
        cand1 = os.path.abspath(os.path.join(cur, '..', 'DATASET'))
        cand2 = os.path.abspath(os.path.join(cur, '..', '..', 'fatformer-xla', 'DATASET'))
        cand3 = os.path.abspath(r"d:\GIT REPO\.nam4\fatformer-xla\DATASET")
        if os.path.exists(cand1):
            dataset_root = cand1
        elif os.path.exists(cand2):
            dataset_root = cand2
        elif os.path.exists(cand3):
            dataset_root = cand3
        else:
            dataset_root = cand1
            os.makedirs(dataset_root, exist_ok=True)

    downloads_dir = os.path.join(dataset_root, 'datasets', 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    logs_dir = os.path.join(dataset_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'download_datasets.log')

    log_msg("=========================================================", log_file)
    log_msg("KHOI DONG TRINH TAI DATASET FATFORMER", log_file)
    log_msg(f"DATASET ROOT: {dataset_root}", log_file)
    log_msg(f"MUC TIEU TAI: {args.target}", log_file)
    log_msg("=========================================================", log_file)

    targets = ['val', 'test_progan', 'test_gans', 'train'] if args.target == 'all' else [args.target]

    for t_key in targets:
        info = DATASET_CATALOG[t_key]
        log_msg(f"\n>>> Xu ly phan: {info['name']} ({t_key})", log_file)
        for item in info['files']:
            filename = item['filename']
            url = item['url']
            expected_size = item.get('expected_size')
            target_path = os.path.join(downloads_dir, filename)

            ok = download_file(url, target_path, expected_size=expected_size, log_file=log_file)
            if ok and not args.no_extract and item.get('extract_to') and filename.endswith('.zip'):
                extract_target = os.path.join(dataset_root, 'datasets', item['extract_to'])
                extract_zip(target_path, extract_target, log_file=log_file)

    log_msg("\nHoan tat toan bo yeu cau tai du lieu!", log_file)


if __name__ == '__main__':
    main()
