"""Small local/Colab UI over the selected, calibrated FatFormer package."""
from __future__ import annotations

import argparse
import csv
import getpass
import html
import io
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision.transforms import Compose

from tools.infer_unified import get_default_transform, load_unified_model
from tools.package_unified_fatformer import compute_file_sha256

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "DATASET/checkpoints/phase7/demo/unified_fatformer_v2.pth"
PACKAGE_SHA = "248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0"
SOURCE_SHA = "5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c"
TEMPERATURE = 2.7832491397857666
THRESHOLD = 0.7918508648872375
MAX_BYTES, MAX_PIXELS = 32 * 1024**2, 25_000_000
EXAMPLE_DIR = ROOT / "DATASET/logs/phase7/task_7.15/current/raw/verification_images"
EXAMPLE_CSV = ROOT / "DATASET/logs/phase7/task_7.15/current/analysis/verification_images.csv"
TRANSFORM = get_default_transform()
SPATIAL = Compose(TRANSFORM.transforms[:2])
NORMALIZE = Compose(TRANSFORM.transforms[2:])
SINGLE, COMPARE = "Phân tích một ảnh", "So sánh JPEG Q30"


def read_image(path):
    if not path:
        raise ValueError("Bạn hãy chọn một ảnh trước khi phân tích.")
    path = Path(path)
    if not path.is_file():
        raise ValueError("Không tìm thấy ảnh. Bạn hãy chọn lại file.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 32 MiB.")
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Chỉ hỗ trợ ảnh JPEG, PNG và WebP.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Ảnh vượt quá giới hạn 25 triệu pixel.")
            if getattr(image, "is_animated", False):
                raise ValueError("Demo chỉ hỗ trợ ảnh tĩnh.")
            # Match the measured PIL->RGB path: no EXIF rotation/alpha compositing.
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Không giải mã được ảnh hoặc file ảnh đã bị hỏng.") from exc


def jpeg_q30(crop):
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def example_records():
    if not EXAMPLE_CSV.is_file():
        return []
    with EXAMPLE_CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["path"] = EXAMPLE_DIR / row["file"]
    return rows


def load_demo_model(checkpoint=CHECKPOINT, device="cpu"):
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    if device not in {"cpu", "cuda"}:
        raise ValueError("Thiết bị phải là cpu hoặc cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Chưa có CUDA. Trên Colab, chọn Runtime > Change runtime type > GPU.")
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu model đã chọn: {path}")
    started = time.perf_counter()
    if compute_file_sha256(str(path)) != PACKAGE_SHA:
        raise ValueError("SHA-256 model không khớp bản đã chọn. Không dùng checkpoint khác cùng tên.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_unified_model(str(path), clip_path="", device=device)
    calibration = metadata["calibration"]
    if (metadata.get("source_checkpoint_sha256") != SOURCE_SHA
            or metadata["layout"]["indices"] != [7, 15, 23]
            or model.router_enabled
            or not math.isclose(calibration["T"], TEMPERATURE, abs_tol=1e-8)
            or not math.isclose(model.fixed_threshold, THRESHOLD, abs_tol=1e-8)):
        raise ValueError("Metadata hoặc calibration không khớp hợp đồng demo.")
    info = dict(package_sha256=PACKAGE_SHA, source_sha256=SOURCE_SHA, device=device,
                precision="FP16 autocast / FP32 softmax" if device == "cuda" else "FP32",
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                temperature=model.clean_temperature, threshold=model.fixed_threshold,
                router_enabled=False, python=platform.python_version(), torch=torch.__version__,
                pillow=Image.__version__, cold_load_ms=(time.perf_counter() - started) * 1000)
    return model, info


def occlusion_explanation(model, crop, probability, device="cpu"):
    """Measure Fake-score sensitivity; this is not a tampering segmentation."""
    started = time.perf_counter()
    tensor = NORMALIZE(crop).unsqueeze(0).to(device)
    rows = []
    # lean: fixed 4x4, batch 1 limits memory; finer localization needs separate validation.
    for row in range(4):
        for col in range(4):
            x0, y0 = col * 56, row * 56
            masked = tensor.clone()
            masked[:, :, y0:y0+56, x0:x0+56] = 0  # CLIP channel means in RGB space.
            with torch.inference_mode(), torch.amp.autocast(
                    device, dtype=torch.float16, enabled=device == "cuda"):
                value = float(model.predict(masked)["prob_fake"][0])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Điểm giải thích vùng ảnh không hợp lệ.")
            rows.append(dict(tile=len(rows)+1, row=row+1, column=col+1,
                             box_xyxy=[x0, y0, x0+56, y0+56], masked_prob_fake=value,
                             delta_fake_pp=(probability-value)*100))
    pixels = np.asarray(crop).astype(np.float32).copy()
    for region in rows:
        x0, y0, x1, y1 = region["box_xyxy"]
        delta = region["delta_fake_pp"]
        color = np.array([204, 67, 47] if delta > 0 else [46, 100, 175])
        alpha = min(abs(delta)/10, 1) * 0.55
        pixels[y0:y1, x0:x1] = pixels[y0:y1, x0:x1]*(1-alpha) + color*alpha
    overlay = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    draw = ImageDraw.Draw(overlay)
    for region in rows:
        x0, y0, x1, y1 = region["box_xyxy"]
        draw.rectangle((x0, y0, x1-1, y1-1), outline="white")
        text = f'{region["tile"]}:{region["delta_fake_pp"]:+.1f}'
        box = draw.textbbox((x0+2, y0+2), text)
        draw.rectangle(box, fill="black")
        draw.text((x0+2, y0+2), text, fill="white")
    return overlay, dict(method="nonoverlapping_occlusion_4x4", target="prob_fake_clean",
                         baseline_prob_fake=probability, replacement_normalized=0,
                         replacement_rgb=list(NORMALIZE.transforms[-1].mean),
                         coordinate_space="224x224 clean center crop; xyxy end exclusive",
                         n_forwards=16, regions=rows, elapsed_ms=(time.perf_counter()-started)*1000,
                         localization_validated=False)


def explanation_markdown(explanation):
    if not explanation:
        return "Bật **Giải thích vùng ảnh**, bấm Phân tích rồi mở mục này để xem kết quả."
    strongest = max(explanation["regions"], key=lambda row: row["delta_fake_pp"])
    finding = (f'Che ô **{strongest["tile"]}** (hàng {strongest["row"]}, cột {strongest["column"]}) '
               f'làm điểm Fake giảm nhiều nhất: **{strongest["delta_fake_pp"]:.4f} điểm phần trăm**.'
               if strongest["delta_fake_pp"] > 0 else "Không ô nào làm điểm Fake giảm khi bị che trong phép thử này.")
    return (f'{finding}\n\n**Đỏ / số dương:** che vùng làm điểm Fake giảm. '
            '**Xanh / số âm:** che vùng làm điểm Fake tăng. Số trên mỗi ô là chênh lệch điểm phần trăm. '
            'Màu đậm dần đến mức 10 điểm phần trăm; số đo không được chuẩn hóa lại theo từng ảnh.\n\n'
            f'Đã chạy thêm **16 lượt** trên crop gốc, mất **{explanation["elapsed_ms"]/1000:.1f} giây**. '
            'Mỗi lượt thay một ô bằng màu trung bình CLIP, giữ các ô khác. '
            '**Đây là độ nhạy với phép che, không phải xác suất vùng giả hay bằng chứng chỉnh sửa.** '
            'Phép che có thể tạo ảnh ngoài phân phối; kết quả phụ thuộc cách che và kích thước ô. '
            'Không suy ra mắt/tóc/tay bị làm giả từ bản đồ này; phần ngoài crop chưa được phân tích.')


def analyze(model, path, device="cpu", compare=False, explain=False):
    """One image/view per forward so CPU/GPU references use the same batch size."""
    started = time.perf_counter()
    image = read_image(path)
    crop = SPATIAL(image)
    views = {"clean": crop}
    if compare:
        views["q30"] = jpeg_q30(crop)
    preprocess_ms = (time.perf_counter() - started) * 1000
    results = {}
    for condition, view in views.items():
        tensor = NORMALIZE(view).unsqueeze(0).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(
                device, dtype=torch.float16, enabled=device == "cuda"):
            output = model.predict(tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - forward_started) * 1000
        p, label, threshold = (output[key][0].item() for key in ("prob_fake", "label", "threshold"))
        if (not math.isfinite(p) or not 0 <= p <= 1 or label not in (0, 1)
                or not math.isclose(threshold, THRESHOLD, abs_tol=1e-7)
                or int(label) != int(p >= threshold)):
            raise ValueError("Output model không hợp lệ hoặc không khớp ngưỡng đã chọn.")
        results[condition] = dict(prob_fake=p, label="FAKE" if label else "REAL",
                                  threshold=threshold, inference_ms=inference_ms)
    digest = compute_file_sha256(str(path))
    source = next((r for r in example_records() if r["sha256"] == digest), None)
    results.update(preprocess_ms=preprocess_ms, total_backend_ms=(time.perf_counter() - started) * 1000,
                   device=device, image_size=list(image.size), batch_size=1,
                   source_label=("FAKE" if int(source["target"]) else "REAL") if source else None,
                   source_id=source["sample_id"] if source else None)
    if explain:
        views["explanation"], results["explanation"] = occlusion_explanation(
            model, crop, results["clean"]["prob_fake"], device)
        results["total_backend_ms"] = (time.perf_counter()-started)*1000
    return image, views, results


def verify_demo(model, info, report_path):
    records = example_records()
    if len(records) != 20:
        raise ValueError("Kiểm chứng cần đúng 20 ảnh và CSV verification của dự án.")
    for record in records:
        if compute_file_sha256(str(record["path"])) != record["sha256"]:
            raise ValueError(f"Ảnh verification sai hash: {record['file']}")
    for _ in range(3):
        analyze(model, records[0]["path"], info["device"])
    timings, errors, rows = [], [], []
    # Same-backend reference, independent of the UI callback and preview split.
    for record in records:
        _, views, actual = analyze(model, record["path"], info["device"], compare=True)
        for condition, view in views.items():
            x = (TRANSFORM(read_image(record["path"])) if condition == "clean"
                 else NORMALIZE(jpeg_q30(SPATIAL(read_image(record["path"]))))).unsqueeze(0).to(info["device"])
            with torch.inference_mode(), torch.amp.autocast(
                    info["device"], dtype=torch.float16, enabled=info["device"] == "cuda"):
                reference = model.predict(x)
            expected = float(reference["prob_fake"][0])
            error = abs(expected - actual[condition]["prob_fake"])
            if error > 1e-6 + 1e-5 * abs(expected) or int(reference["label"][0]) != int(actual[condition]["label"] == "FAKE"):
                raise ValueError(f"Demo/reference differ: {record['file']} {condition}")
            errors.append(error)
        timings.append(actual["total_backend_ms"])
        rows.append(dict(file=record["file"], target=int(record["target"]), **actual))
        print(f"Verified {len(rows)}/20: {record['file']}", flush=True)
    if info["device"] == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_gib = None
    report = dict(schema="fatformer_demo_verification/v1", environment=info,
                  n_images=20, n_views=40, warmup_images=3, same_backend_reference=True,
                  max_probability_error=max(errors), backend_compare_p50_ms=float(np.median(timings)),
                  backend_compare_p95_ms=float(np.percentile(timings, 95)), cuda_peak_allocated_gib=peak_gib,
                  note="Clean/Q30 pair timing; excludes queue/browser. Does not establish G7.3 or GPU/CPU bit equivalence.",
                  results=rows)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


CSS = """
/* NỀN TRẮNG TỔNG THỂ */
body, html, .gradio-container, .dark body, body.dark, .dark .gradio-container, .gradio-container.dark {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.gradio-container {max-width: 1200px !important; margin: auto !important}
.demo-header {padding: 14px 18px; border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03)}
.demo-kicker {font: 700 11px/1.5 Consolas, monospace; letter-spacing: .12em; color: #000000; text-transform: uppercase}
.demo-header h1 {font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin: 6px 0; color: #000000}
.demo-header p {color: #111827; max-width: 820px; font-size: 14px; margin: 4px 0 10px; line-height: 1.45}

/* CHỮ TRẮNG NỀN ĐEN: Status, Primary Button, Block Label Badges */
.status {
    display: inline-block;
    background: #000000 !important;
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
}
.gradio-container button.primary,
.gradio-container button.primary * {
    background: #000000 !important;
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid #000000 !important;
    fill: #ffffff !important;
}
.gradio-container button.primary:hover {
    background: #1f2937 !important;
    background-color: #1f2937 !important;
}
.gradio-container .block-label,
.gradio-container span.block-label,
.gradio-container label.block-label,
.gradio-container span[data-testid="block-info"],
.gradio-container .block-label *,
.gradio-container label.block-label *,
.gradio-container span[data-testid="block-info"] * {
    background: #000000 !important;
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    fill: #ffffff !important;
}

/* CHỮ ĐEN NỀN TRẮNG: Blocks, Upload area, Accordions, Examples, Footnotes */
.gradio-container .block,
.gradio-container.dark .block {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}
.gradio-container .upload-container,
.gradio-container [data-testid="file-upload"],
.gradio-container .empty.file-upload-holder,
.gradio-container .file-preview {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 2px dashed #94a3b8 !important;
}
.gradio-container .upload-container *,
.gradio-container [data-testid="file-upload"] *,
.gradio-container .empty.file-upload-holder * {
    color: #000000 !important;
    font-weight: 600 !important;
}
.gradio-container .upload-container svg,
.gradio-container [data-testid="file-upload"] svg {
    stroke: #000000 !important;
    fill: #000000 !important;
}
.gradio-container .accordion,
.gradio-container .accordion > .label-wrap,
.gradio-container details,
.gradio-container details > summary,
.gradio-container .label-wrap {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}
.gradio-container .accordion > .label-wrap *,
.gradio-container details > summary *,
.gradio-container .label-wrap span,
.gradio-container .label-wrap svg {
    color: #000000 !important;
    font-weight: 700 !important;
    stroke: #000000 !important;
}
.gradio-container .examples-title,
.gradio-container .examples-holder,
.gradio-container .examples-holder *,
.gradio-container [data-testid="examples"] *,
.gradio-container [data-testid="examples"] label span {
    color: #000000 !important;
    font-weight: 700 !important;
}
.gradio-container button.gallery-item,
.gradio-container .examples-table button,
.gradio-container .table-wrap button {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    text-align: left !important;
    cursor: pointer !important;
}
.gradio-container button.gallery-item:hover,
.gradio-container .examples-table button:hover,
.gradio-container .table-wrap button:hover {
    background: #f1f5f9 !important;
    border-color: #000000 !important;
}
.gradio-container .paginate,
.gradio-container .paginate button {
    color: #000000 !important;
    font-weight: 700 !important;
}
.gradio-container fieldset,
.gradio-container .gr-check-radio,
.gradio-container .wrap.default,
.gradio-container .checkbox-group label,
.gradio-container .radio-group label {
    background: #ffffff !important;
    color: #000000 !important;
}
.gradio-container fieldset span,
.gradio-container .checkbox-group span,
.gradio-container .radio-group span,
.gradio-container label.checkbox-label span,
.gradio-container input[type="radio"] + span,
.gradio-container input[type="checkbox"] + span {
    color: #000000 !important;
    font-weight: 600 !important;
}
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .demo-footnote {
    color: #000000 !important;
}
.gradio-container button.secondary,
.gradio-container button.secondary * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #94a3b8 !important;
    font-weight: 700 !important;
}
.gradio-container button.secondary:hover {
    background: #f1f5f9 !important;
}
.result-empty {
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: #ffffff !important;
    border: 2px dashed #94a3b8;
    border-radius: 8px;
    padding: 20px;
    color: #000000 !important;
    text-align: center;
}
.result-empty b {font-size: 18px; color: #000000 !important; margin-bottom: 4px}
.result-empty p {color: #111827 !important}
.result {padding: 14px 16px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03)}
.result-label {font-size: 22px; font-weight: 700; color: #14532d; background: #dcfce7; border: 1px solid #86efac; border-radius: 6px; padding: 6px 12px; margin: 6px 0 12px; display: inline-block}
.fake-label {color: #7f1d1d !important; background: #fee2e2 !important; border-color: #fca5a5 !important}
.score-row {display: flex; justify-content: space-between; align-items: baseline}
.score-value {font: 700 28px Consolas, monospace; color: #000000}
.score-name {font-size: 13px; font-weight: 700; color: #000000}
.score-track {height: 10px; background: #e2e8f0; border-radius: 5px; position: relative; margin: 10px 0 8px; border: 1px solid #cbd5e1}
.score-fill {height: 100%; background: #1d4ed8; border-radius: 4px; transition: width 0.3s ease}
.fake-fill {background: #b91c1c !important}
.real-fill {background: #15803d !important}
.score-threshold {position: absolute; top: -4px; height: 18px; width: 2.5px; background: #000000}
.score-note {font-size: 12px; font-weight: 600; color: #111827}
.source-note {padding: 8px 12px; font-size: 12.5px; color: #000000; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 10px}
.demo-error {border-left: 4px solid #b91c1c; padding: 12px 14px; color: #7f1d1d; background: #fef2f2; border-radius: 4px; font-weight: 500}
.decision-reason {padding: 10px 12px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #000000; line-height: 1.45}
.decision-reason b {color: #000000}
.decision-reason p {margin: 4px 0}
.comparison-note {padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; font-size: 13px; color: #000000; margin-top: 10px}
.comparison-note b {color: #000000}
.view-label {font-size: 13px !important; font-weight: 700 !important; color: #000000 !important}
.demo-footnote {font-size: 12px !important; color: #111827 !important; line-height: 1.4 !important}
footer {display: none !important}
"""
EMPTY_RESULT = '<div class="result-empty"><b>Kết quả sẽ xuất hiện ở đây</b><p>Chọn ảnh và bấm Phân tích để chạy model.</p></div>'
CURATED_EXAMPLE_INDICES = [2, 12, 6, 8, 14, 0, 18, 1, 10, 17]
CURATED_LABELS = [
    "[FAKE] Midjourney v5.2 (Thương mại)",
    "[REAL] Chân dung FFHQ (Ảnh thật)",
    "[FAKE] SDXL LCM-LoRA (Latent Diffusion)",
    "[REAL] Máy ảnh RAISE (Ảnh chụp)",
    "[FAKE] DFGAN (Modern GAN)",
    "[REAL] Đời sống COCO (Ảnh thật)",
    "[FAKE] Stable Cascade (Autoregressive)",
    "[REAL] Đồ vật LAION (Ảnh thật)",
    "[FAKE] Hourglass (Pixel Diffusion)",
    "[REAL] Phong cảnh LAION (Ảnh thật)",
]


def result_html(result):
    source = result.get("source_label")
    blocks = []
    for key, title in (("clean", "Kết luận trên crop gốc"), ("q30", "Kết luận trên crop JPEG Q30")):
        if key not in result:
            continue
        row = result[key]
        score, mark = row["prob_fake"] * 100, row["threshold"] * 100
        label_class = " fake-label" if row["label"] == "FAKE" else ""
        fill_class = " fake-fill" if row["label"] == "FAKE" else " real-fill"
        is_fake = row["label"] == "FAKE"
        interpretation = "ảnh tổng hợp theo model" if is_fake else "ảnh thật theo model"
        relation = "≥" if is_fake else "<"
        direction = "cao hơn hoặc bằng" if is_fake else "thấp hơn"
        blocks.append(f'<section class="result"><div class="score-name">{title}</div>'
                      f'<div class="result-label{label_class}">Dự đoán {row["label"]} · {interpretation}</div>'
                      f'<div class="score-row"><span class="score-name">Điểm Fake sau calibration</span>'
                      f'<span class="score-value">{score:.2f}%</span></div>'
                      f'<div class="score-track"><div class="score-fill{fill_class}" style="width:{score:.4f}%"></div>'
                      f'<span class="score-threshold" style="left:{mark:.4f}%"></span></div>'
                      f'<div class="score-note">Ngưỡng phân loại: {mark:.6f}%</div>'
                      f'<div class="decision-reason"><b>Vì sao có quyết định này?</b>'
                      f'<p>Điểm Fake <b>{score:.6f}% {relation} {mark:.6f}%</b>. '
                      f'Điểm {direction} ngưỡng nên model trả nhãn <b>{row["label"]}</b>.</p>'
                      f'<p>Khoảng cách tới ngưỡng: <b>{abs(score-mark):.6f} điểm phần trăm</b>. '
                      'Đây không phải mức độ chắc chắn rằng model đúng.</p>'
                      '<p>Quy tắc cố định: FAKE khi điểm Fake ≥ ngưỡng; REAL khi thấp hơn. '
                      'So sánh dùng số đầy đủ, các số hiển thị được làm tròn.</p></div></section>')
    if "q30" in result:
        clean, q30 = result["clean"], result["q30"]
        delta = (q30["prob_fake"]-clean["prob_fake"])*100
        change = "Nhãn thay đổi sau nén" if q30["label"] != clean["label"] else "Nhãn giữ nguyên sau nén"
        blocks.append(f'<div class="comparison-note"><b>{change}: {clean["label"]} → {q30["label"]}.</b> '
                      f'Điểm Fake Q30 − gốc: <b>{delta:+.4f} điểm phần trăm</b>. '
                      'Đây là phản ứng của model với nén JPEG trên ảnh này, không xác nhận ảnh đã bị làm giả.</div>')
    if source:
        blocks.append(f'<div class="source-note">Nhãn nguồn của ảnh mẫu: <b>{source}</b> · Được ghi độc lập với dự đoán.</div>')
    return "".join(blocks)


def build_ui(model, info):
    import gradio as gr

    def preview(path):
        if not path:
            return None, None, None, EMPTY_RESULT, "", None, explanation_markdown(None), {}
        try:
            image = read_image(path)
            digest = compute_file_sha256(str(path))
            source = next((r for r in example_records() if r["sha256"] == digest), None)
            if source:
                target_str = "FAKE (ảnh tổng hợp)" if int(source["target"]) else "REAL (ảnh thật)"
                preview_note = (
                    f'<div class="result-empty" style="border: 1px solid #cbd5e1; background: #f8fafc !important;">'
                    f'<b>Đã nạp ảnh mẫu: {html.escape(source["file"])}</b>'
                    f'<p style="margin: 6px 0; color: #000000;">Nhãn kiểm định: <b>{target_str}</b> · Nhóm: <code>{html.escape(source["sample_id"].split("/")[0])}</code></p>'
                    f'<p style="color: #374151; font-size: 13px;">Bấm nút <b>🔍 Phân tích ảnh</b> bên dưới để chạy mô hình FatFormer XLA.</p>'
                    f'</div>'
                )
            else:
                preview_note = EMPTY_RESULT
            return image, SPATIAL(image), None, preview_note, "", None, explanation_markdown(None), {}
        except ValueError as exc:
            return None, None, None, '<div class="demo-error">'+html.escape(str(exc))+"</div>", "", None, explanation_markdown(None), {}

    def predict(path, mode, explain=False):
        try:
            image, views, result = analyze(model, path, info["device"], mode == COMPARE, explain)
            detail = dict(**result, precision=info["precision"], package_sha256=PACKAGE_SHA)
            timing = (f'⏱️ Backend **{result["total_backend_ms"]:.0f} ms** · '
                      f'{info["device"].upper()} / {info["precision"]} · Không gồm thời gian chờ queue/trình duyệt')
            return (image, views["clean"], views.get("q30"), result_html(result), timing,
                    views.get("explanation"), explanation_markdown(result.get("explanation")), detail)
        except (ValueError, torch.cuda.OutOfMemoryError) as exc:
            message = ("GPU không đủ bộ nhớ cho ảnh này. Hãy khởi động lại runtime hoặc dùng Windows CPU."
                       if isinstance(exc, torch.cuda.OutOfMemoryError) else str(exc))
            return None, None, None, '<div class="demo-error">'+html.escape(message)+"</div>", "", None, explanation_markdown(None), {}

    with gr.Blocks(title="FatFormer XLA · Phân tích ảnh", analytics_enabled=False, delete_cache=(600, 600)) as app:
        gr.HTML('<header class="demo-header"><div class="demo-kicker">FATFORMER XLA / DEMO NGHIÊN CỨU</div>'
                '<h1>Kiểm tra ảnh tổng hợp</h1><p>Quan sát cách model phân tích một khung hình — '
                'từ vùng ảnh đầu vào đến điểm Fake sau calibration.</p>'
                f'<span class="status">Model sẵn sàng · {html.escape(info["device"].upper())} · {html.escape(info["precision"])}</span></header>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                file = gr.File(label="Tải ảnh lên để phân tích", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                records = example_records()
                if records and len(records) >= 20 and all(records[i]["path"].is_file() for i in CURATED_EXAMPLE_INDICES):
                    curated = [records[i] for i in CURATED_EXAMPLE_INDICES]
                    gr.Examples(
                        examples=[[str(r["path"])] for r in curated],
                        inputs=file,
                        example_labels=CURATED_LABELS,
                        label="Ảnh mẫu kiểm định · 5 họ sinh ảnh & 5 nguồn ảnh thật (Click để chọn)",
                        examples_per_page=5,
                        cache_examples=False,
                    )
                elif records and all(row["path"].is_file() for row in records[:4]):
                    gr.Examples([[str(r["path"])] for r in records[:4]], inputs=file, label="Ảnh mẫu có nguồn · chạy model khi bấm Phân tích", cache_examples=False)
                original = gr.Image(label="Ảnh đầu vào (Xem trước)", interactive=False, height=270, format="png")
                mode = gr.Radio([SINGLE, COMPARE], value=SINGLE, label="Chế độ phân tích")
                explain = gr.Checkbox(value=False, label="Giải thích vùng ảnh · thêm 16 lượt chạy trên crop gốc")
                gr.Markdown("Bản đồ cho biết vùng ảnh ảnh hưởng đến điểm Fake khi bị che. "
                            "Chưa xác định được vùng giả mạo; chế độ này chậm hơn, đặc biệt trên CPU.", elem_classes="demo-footnote")
                with gr.Row():
                    submit = gr.Button("🔍 Phân tích ảnh", variant="primary", scale=3)
                    clear = gr.Button("Xóa", variant="secondary", scale=1)
            with gr.Column(scale=6):
                verdict = gr.HTML(EMPTY_RESULT)
                with gr.Row():
                    clean = gr.Image(label="Crop model nhận · 224 × 224", interactive=False, height=220, format="png")
                    q30 = gr.Image(label="Crop nén JPEG Q30", interactive=False, height=220, format="png")
                timing = gr.Markdown("")
                with gr.Accordion("Vùng ảnh ảnh hưởng đến điểm Fake (Occlusion Sensitivity)", open=False):
                    influence = gr.Image(label="Độ nhạy trên crop gốc · không phải mặt nạ giả mạo", interactive=False, height=300, format="png")
                    explanation = gr.Markdown(explanation_markdown(None))
                gr.Markdown("Kết quả là dự đoán của mô hình trên vùng crop hiển thị. "
                            "Điểm Fake không phải độ chính xác của một ảnh cụ thể. "
                            "JPEG Q30 được tạo sau crop để so sánh có kiểm soát.", elem_classes="demo-footnote")
                with gr.Accordion("Thông số kỹ thuật và kết quả chi tiết", open=False):
                    gr.JSON(info, label="Phiên model")
                    detail = gr.JSON({}, label="Lần phân tích")
        outputs = [original, clean, q30, verdict, timing, influence, explanation, detail]
        file.change(preview, file, outputs, queue=False, api_visibility="private")
        mode.change(preview, file, outputs, queue=False, api_visibility="private")
        explain.change(preview, file, outputs, queue=False, api_visibility="private")
        submit.click(predict, [file, mode, explain], outputs, api_name="analyze", concurrency_limit=1)
        clear.click(lambda: (None, *preview(None)), outputs=[file, *outputs], queue=False, api_visibility="private")
    return app.queue(max_size=4, default_concurrency_limit=1)


def launch_demo(app, *, host="127.0.0.1", port=7860, share=False, auth=None, prevent_thread_lock=False):
    import gradio as gr
    if share and not auth:
        raise ValueError("Link chia sẻ cần tài khoản truy cập cho phiên demo.")
    theme = gr.themes.Base(
        font=("Segoe UI", "Arial", "sans-serif"),
        font_mono=("Consolas", "monospace")
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        body_text_color="#000000",
        body_text_color_dark="#000000",
        body_text_color_subdued="#111827",
        body_text_color_subdued_dark="#111827",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_label_background_fill="#000000",
        block_label_background_fill_dark="#000000",
        block_label_text_color="#ffffff",
        block_label_text_color_dark="#ffffff",
        block_label_text_weight="700",
        accordion_text_color="#000000",
        accordion_text_color_dark="#000000",
        button_primary_background_fill="#000000",
        button_primary_background_fill_dark="#000000",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color="#000000",
        button_secondary_text_color_dark="#000000",
        button_secondary_border_color="#94a3b8",
        button_secondary_border_color_dark="#94a3b8",
    )
    return app.launch(server_name=host, server_port=port, share=share, auth=auth,
                      prevent_thread_lock=prevent_thread_lock, max_file_size=MAX_BYTES,
                      allowed_paths=[str(EXAMPLE_DIR)],
                      blocked_paths=[str(ROOT / p) for p in (".env", ".git", "DATASET/checkpoints", "wait", ".venv-demo")],
                      css=CSS, theme=theme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Run 20 real clean/Q30 comparisons without opening UI")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    model, info = load_demo_model(args.checkpoint, args.device)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.verify_only:
        path = args.report or str(ROOT / f"logs/demo/{args.device}_verification.json")
        report = verify_demo(model, info, path)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
        return
    auth = None
    if args.share:
        auth = (input("Tên truy cập phiên demo: ").strip(), getpass.getpass("Mật khẩu phiên demo: "))
        if not all(auth):
            parser.error("Tên truy cập và mật khẩu không được rỗng.")
    launch_demo(build_ui(model, info), host=args.host, port=args.port, share=args.share, auth=auth)


if __name__ == "__main__":
    main()
