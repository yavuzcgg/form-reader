"""
PaddleOCR Test - TÜBİTAK 3501 Projesi
======================================
Eğitilmiş PaddleOCR modelini sentetik veri üzerinde test eder.
Subprocess ile çalışır - paket çakışması olmaz.

Kullanım:
    python src/paddle_test.py [model_dir]
"""

import os
import sys
import csv
import random
import unicodedata
import subprocess


def test_paddle_model(
    model_dir=None,
    data_dir=None,
    csv_path=None,
    samples_per_category=5,
    seed=42
):
    """
    PaddleOCR modelini subprocess ile test eder.
    Pandas/matplotlib bağımlılığı YOK.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    if data_dir is None:
        data_dir = os.path.join(PROJECT_DIR, "data", "synthetic")
    if csv_path is None:
        csv_path = os.path.join(data_dir, "metadata.csv")
    if model_dir is None:
        model_dir = os.path.join(PROJECT_DIR, "models", "paddle-turkish-handwritten", "deney006")

    dict_path = os.path.join(PROJECT_DIR, "data", "turkish_dict.txt")

    print(f"Model: {model_dir}")
    print(f"Dict: {dict_path}")
    print(f"Veri: {data_dir}")

    # CSV'den kategorilere göre örnekler seç (pandas olmadan)
    categories = ['number', 'turkish_text', 'date', 'mixed', 'code']
    samples = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows_by_cat = {}
        for row in reader:
            cat = row['category']
            if cat in categories:
                if cat not in rows_by_cat:
                    rows_by_cat[cat] = []
                rows_by_cat[cat].append(row)

    random.seed(seed)
    test_items = []
    for cat in categories:
        if cat in rows_by_cat:
            cat_rows = rows_by_cat[cat]
            selected = random.sample(cat_rows, min(samples_per_category, len(cat_rows)))
            for row in selected:
                test_items.append({
                    'category': cat,
                    'file_name': row['file_name'],
                    'text': unicodedata.normalize('NFC', str(row['text']).strip())
                })

    # Test scriptini subprocess ile çalıştır (PaddleOCR import sorunsuz)
    test_script = _generate_test_script(model_dir, dict_path, data_dir, test_items)

    # NVIDIA DLL'lerini subprocess PATH'ine ekle (paddle import öncesi gerekli)
    import site
    sp = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else ''
    base = os.path.join(sys.prefix, "Lib", "site-packages")
    nvidia_bins = [
        os.path.join(base, "nvidia", d, "bin")
        for d in ["cudnn", "cuda_runtime", "cublas", "cufft", "cusolver", "cusparse", "nvjitlink"]
    ]
    extra_path = os.pathsep.join([d for d in nvidia_bins if os.path.exists(d)])

    env = os.environ.copy()
    env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=PROJECT_DIR,
        env=env
    )

    print(result.stdout)
    if result.stderr:
        # Sadece gerçek hataları göster (warning'leri filtrele)
        errors = [l for l in result.stderr.split('\n')
                  if 'Error' in l or 'Exception' in l or 'Traceback' in l]
        if errors:
            print("[STDERR]", '\n'.join(errors[:5]))


def _generate_test_script(model_dir, dict_path, data_dir, test_items):
    """Subprocess'te çalışacak test scriptini string olarak üretir."""

    # Test öğelerini string olarak serialize et
    items_str = repr(test_items)

    script = f'''
import os
import sys
import unicodedata

# NVIDIA DLL'lerini PATH'e ekle (PaddlePaddle 3.0 cuDNN/CUDA)
nvidia_dirs = [
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cufft", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cusolver", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cusparse", "bin"),
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "nvjitlink", "bin"),
]
for d in nvidia_dirs:
    if os.path.exists(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(d)

from paddleocr import PaddleOCR

model_dir = r"{model_dir}"
dict_path = r"{dict_path}"
data_dir = r"{data_dir}"

ocr = PaddleOCR(
    text_recognition_model_dir=model_dir,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

test_items = {items_str}

print("=" * 60)
print("SENTETIK VERI TESTI (PaddleOCR - Deney #006)")
print("=" * 60)

correct = 0
total = 0
current_cat = ""

for item in test_items:
    cat = item['category']
    if cat != current_cat:
        print(f"\\n--- {{cat.upper()}} ---")
        current_cat = cat

    img_path = os.path.join(data_dir, item['file_name'])
    expected = item['text']

    try:
        result = ocr.ocr(img_path, det=False, cls=False)
        if result and result[0]:
            predicted = result[0][0][0]
            confidence = result[0][0][1]
        else:
            predicted = ""
            confidence = 0.0
    except Exception as e:
        predicted = f"[HATA: {{e}}]"
        confidence = 0.0

    predicted = predicted.strip()
    match = "+" if predicted == expected else "X"
    if match == "+":
        correct += 1
    total += 1

    print(f"{{match}} Beklenen: {{expected:<25}} | Tahmin: {{predicted:<25}} | Guven: {{confidence:.2f}}")

print("\\n" + "=" * 60)
if total > 0:
    print(f"DOGRULUK: {{correct}}/{{total}} = {{100*correct/total:.1f}}%")
else:
    print("Hic test yapilmadi!")
print("=" * 60)
'''
    return script


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    test_paddle_model(model_dir=model_dir)
