"""
PaddleOCR Full Pipeline - TÜBİTAK 3501 Projesi
================================================
Gerçek form okuma: Detection (metin bul) + Recognition (metin oku)

Çizgi bazlı DEĞİL, metin bazlı detection:
- Silik çizgiler sorun DEĞİL
- Paddle Inference API ile detection (DLL çakışması yok)
- subprocess ile recognition (fine-tuned Türkçe model)

Pipeline:
    1. detect_text_boxes() → Paddle Inference API ile metin kutularını bul
    2. crop_boxes() → Her kutuyu resimden kes, geçici dosyaya kaydet
    3. recognize_text() → subprocess + infer_rec.py ile her kutuyu oku
    4. Sonuçları birleştir

Kullanım:
    python src/paddle_pipeline.py data/raw/S25C-925103023031.jpg
"""

import os
import sys
import subprocess
import tempfile
import time
import numpy as np
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PADDLEOCR_DIR = os.path.join(PROJECT_DIR, "PaddleOCR")

# Model paths
DET_MODEL_DIR = os.path.join(PADDLEOCR_DIR, "pretrained_models", "ch_PP-OCRv4_det_infer")
REC_CONFIG = os.path.join(PROJECT_DIR, "configs", "turkish_rec.yml")


def _setup_dll():
    """NVIDIA DLL'lerini yükle (Windows PaddlePaddle 3.0 için gerekli)."""
    base = os.path.join(sys.prefix, "Lib", "site-packages")
    for d in ["cudnn", "cuda_runtime", "cublas", "cufft", "cusolver", "cusparse", "nvjitlink"]:
        p = os.path.join(base, "nvidia", d, "bin")
        if os.path.exists(p):
            os.add_dll_directory(p)
    paddle_libs = os.path.join(base, "paddle", "libs")
    if os.path.exists(paddle_libs):
        os.add_dll_directory(paddle_libs)


def detect_text_boxes(image_path, det_limit_side_len=960, det_threshold=0.3):
    """
    Paddle Inference API ile metin kutularını tespit et.
    DLL çakışması YOK - paddle.inference direkt kullanılıyor.

    Args:
        image_path: Form resmi yolu
        det_limit_side_len: Maksimum kenar uzunluğu (küçük = hızlı)
        det_threshold: Detection eşik değeri (0-1)

    Returns:
        list of tuples: [(x, y, w, h), ...] orijinal koordinatlarda
    """
    _setup_dll()
    from paddle.inference import Config, create_predictor

    # Model yükle (CPU - stabil, hızlı yeterli)
    config = Config(
        os.path.join(DET_MODEL_DIR, "inference.pdmodel"),
        os.path.join(DET_MODEL_DIR, "inference.pdiparams")
    )
    config.disable_gpu()
    predictor = create_predictor(config)

    # Resmi oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"[HATA] Resim okunamadı: {image_path}")
        return []

    h, w = img.shape[:2]

    # Resize (det_limit_side_len'e göre)
    ratio = det_limit_side_len / max(h, w)
    new_h = (int(h * ratio) // 32) * 32
    new_w = (int(w * ratio) // 32) * 32
    resized = cv2.resize(img, (new_w, new_h))

    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_data = resized.astype(np.float32) / 255.0
    input_data = (input_data - mean) / std
    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]

    # Inference
    input_handle = predictor.get_input_handle("x")
    input_handle.reshape(list(input_data.shape))
    input_handle.copy_from_cpu(input_data.astype(np.float32))
    predictor.run()

    output_handle = predictor.get_output_handle(predictor.get_output_names()[0])
    output = output_handle.copy_to_cpu()

    # Threshold ile kutuları bul
    mask = (output[0, 0] > det_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Orijinal koordinatlara dönüştür ve filtrele
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > 10 and bh > 5:  # Minimum boyut filtresi
            # Orijinal koordinatlara dönüştür
            ox = int(x / ratio)
            oy = int(y / ratio)
            obw = int(bw / ratio)
            obh = int(bh / ratio)
            # Biraz padding ekle (kenar karakterlerin kesilmemesi için)
            pad = 5
            ox = max(0, ox - pad)
            oy = max(0, oy - pad)
            obw = min(w - ox, obw + 2 * pad)
            obh = min(h - oy, obh + 2 * pad)
            boxes.append((ox, oy, obw, obh))

    # Yukarıdan aşağı, soldan sağa sırala
    boxes.sort(key=lambda b: (b[1], b[0]))

    return boxes


def recognize_text(image_path):
    """
    subprocess + infer_rec.py ile metin oku.
    Çalışan pattern (Hücre 7/8'den kanıtlanmış).

    Returns:
        tuple: (predicted_text, confidence)
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        os.path.join(PADDLEOCR_DIR, "tools", "infer_rec.py"),
        "-c", REC_CONFIG,
        "-o", f"Global.infer_img={image_path}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_DIR,
        env=env,
    )

    predicted = ""
    confidence = 0.0
    for line in result.stdout.split("\n"):
        if "result:" in line:
            try:
                parts = line.split("result:")[1].strip().split("\t")
                predicted = parts[0].strip()
                if len(parts) > 1:
                    confidence = float(parts[1].strip())
            except:
                pass

    return predicted, confidence


def process_form(image_path, det_limit_side_len=960, det_threshold=0.3, max_boxes=50):
    """
    Bir formu oku: Detection → Crop → Recognition.

    Args:
        image_path: Form resmi yolu
        det_limit_side_len: Detection çözünürlüğü
        det_threshold: Detection eşik değeri
        max_boxes: Maksimum kutu sayısı

    Returns:
        list of dict: [{"text": "...", "box": (x,y,w,h), "confidence": 0.95}, ...]
    """
    form_name = os.path.basename(image_path)
    print(f"\n[Pipeline] {form_name}")

    # 1. Detection
    t0 = time.time()
    boxes = detect_text_boxes(image_path, det_limit_side_len, det_threshold)
    t_det = time.time() - t0
    print(f"  [Detection] {len(boxes)} metin kutusu bulundu ({t_det:.1f}s)")

    if not boxes:
        print("  [UYARI] Hiç metin kutusu bulunamadı!")
        return []

    # Çok fazla kutu varsa sınırla
    boxes = boxes[:max_boxes]

    # 2. Crop + Recognition
    img = cv2.imread(image_path)
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="formreader_")

    t0 = time.time()
    for i, (x, y, w, h) in enumerate(boxes):
        # Kutuyu kes
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # Geçici dosyaya kaydet
        crop_path = os.path.join(tmp_dir, f"box_{i:03d}.jpg")
        cv2.imwrite(crop_path, crop)

        # Recognition
        text, conf = recognize_text(crop_path)

        if text:  # Boş olmayan sonuçları ekle
            results.append({
                "text": text,
                "box": (x, y, w, h),
                "confidence": conf,
            })

    t_rec = time.time() - t0
    print(f"  [Recognition] {len(results)} metin okundu ({t_rec:.1f}s)")

    # Geçici dosyaları temizle
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Sonuçları yazdır
    for r in results:
        x, y, w, h = r["box"]
        print(f"  ({x:4d},{y:4d}) {r['text']:<35} güven:{r['confidence']:.2f}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        results = process_form(image_path)
        print(f"\nToplam: {len(results)} metin okundu")
    else:
        # Varsayılan test
        test_form = os.path.join(PROJECT_DIR, "data", "raw", "S25C-925103023031.jpg")
        if os.path.exists(test_form):
            results = process_form(test_form)
        else:
            print("Kullanım: python src/paddle_pipeline.py <form_resmi.jpg>")
