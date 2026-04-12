"""
PaddleOCR Full Pipeline - TÜBİTAK 3501 Projesi
================================================
Gerçek form okuma: Detection (metin bul) + Recognition (metin oku)

Çizgi bazlı değil, METİN bazlı detection:
- Silik çizgiler sorun DEĞİL
- PaddleOCR DBNet ile metin kutularını bulur
- Fine-tuned recognition modeli ile Türkçe okur

Kullanım:
    python src/paddle_pipeline.py data/raw/S25C-925103023031.jpg
"""

import os
import sys
import subprocess
import json


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PADDLEOCR_DIR = os.path.join(PROJECT_DIR, "PaddleOCR")


def run_ocr_on_form(
    image_path,
    rec_model_dir=None,
    det_model_dir=None,
    dict_path=None,
    output_dir=None,
):
    """
    Bir form resmini PaddleOCR full pipeline ile oku.
    Detection (metin bul) + Recognition (metin oku).

    Returns:
        list of dict: [{"text": "...", "box": [[x1,y1], ...], "confidence": 0.95}, ...]
    """

    if rec_model_dir is None:
        rec_model_dir = os.path.join(PROJECT_DIR, "models", "paddle-turkish-handwritten", "deney006")
    if dict_path is None:
        dict_path = os.path.join(PROJECT_DIR, "data", "turkish_dict.txt")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "output", "pipeline_results")

    os.makedirs(output_dir, exist_ok=True)

    # PaddleOCR predict_system.py ile full pipeline
    cmd = [
        sys.executable,
        os.path.join(PADDLEOCR_DIR, "tools", "infer", "predict_system.py"),
        "--image_dir", image_path,
        "--det_model_dir", os.path.join(PADDLEOCR_DIR, "pretrained_models", "ch_PP-OCRv4_rec_server_train"),
        "--rec_model_dir", os.path.join(rec_model_dir, "best_accuracy"),
        "--rec_char_dict_path", dict_path,
        "--use_space_char", "true",
        "--use_angle_cls", "false",
        "--use_gpu", "true",
        "--output", output_dir,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[Pipeline] Form okunuyor: {os.path.basename(image_path)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=PROJECT_DIR,
        env=env,
    )

    # Sonuçları parse et
    results = []
    for line in result.stdout.split('\n'):
        if 'result:' in line.lower():
            print(f"  {line.strip()}")

    if result.returncode != 0:
        # Hata varsa stderr'den son kısmı göster
        err = result.stderr[-500:] if result.stderr else ""
        print(f"[HATA] {err}")

    return result.stdout


def run_simple_test(image_path):
    """
    Tek bir form resmini basitçe test et.
    predict_system.py kullanmadan, detect + rec ayrı ayrı.
    """

    rec_model = os.path.join(PROJECT_DIR, "models", "paddle-turkish-handwritten", "deney006", "best_accuracy")
    dict_path = os.path.join(PROJECT_DIR, "data", "turkish_dict.txt")
    config_path = os.path.join(PROJECT_DIR, "configs", "turkish_rec.yml")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Sadece recognition testi (detection olmadan, tüm resmi oku)
    cmd = [
        sys.executable,
        os.path.join(PADDLEOCR_DIR, "tools", "infer_rec.py"),
        "-c", config_path,
        "-o", f"Global.infer_img={image_path}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=PROJECT_DIR,
        env=env,
    )

    predicted = ""
    confidence = 0.0
    for line in result.stdout.split('\n'):
        if 'result:' in line:
            try:
                parts = line.split('result:')[1].strip().split('\t')
                predicted = parts[0].strip()
                if len(parts) > 1:
                    confidence = float(parts[1].strip())
            except:
                pass

    return predicted, confidence


def test_all_forms(raw_dir=None, max_forms=5):
    """
    data/raw/ klasöründeki formları PaddleOCR ile test et.
    Her form için detection + recognition çalıştır.
    """
    if raw_dir is None:
        raw_dir = os.path.join(PROJECT_DIR, "data", "raw")

    import glob
    forms = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))[:max_forms]

    print("=" * 70)
    print(f"GERÇEK FORM TESTİ - {len(forms)} form")
    print("=" * 70)

    for form_path in forms:
        print(f"\n{'─' * 70}")
        print(f"Form: {os.path.basename(form_path)}")
        print(f"{'─' * 70}")
        run_ocr_on_form(form_path)


def test_processed_boxes(processed_dir=None, categories=None, max_per_cat=5):
    """
    data/processed/ klasöründeki kesilmiş kutuları test et.
    Sadece recognition (detection yok, kutular zaten kesilmiş).
    """
    if processed_dir is None:
        processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
    if categories is None:
        categories = [
            "input_durus_dakika",
            "input_durus_aciklamalar",
            "input_hat_sorumlusu",
            "input_durus_suresi",
            "input_giren_preform",
            "input_etiket_fire",
        ]

    import glob

    print("=" * 70)
    print("GERÇEK FORM KUTULARI TESTİ (PaddleOCR Recognition)")
    print("=" * 70)

    for cat in categories:
        cat_dir = os.path.join(processed_dir, cat)
        if not os.path.exists(cat_dir):
            continue

        images = sorted(glob.glob(os.path.join(cat_dir, "*.jpg")))[:max_per_cat]
        if not images:
            continue

        print(f"\n--- {cat.upper()} ({len(images)} örnek) ---")

        for img_path in images:
            predicted, confidence = run_simple_test(img_path)
            img_name = os.path.basename(img_path)
            print(f"  {img_name:<40} → {predicted:<30} (güven: {confidence:.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Tek form testi
        image_path = sys.argv[1]
        if os.path.isdir(image_path):
            test_all_forms(image_path)
        else:
            run_ocr_on_form(image_path)
    else:
        # Varsayılan: processed kutuları test et
        test_processed_boxes()
