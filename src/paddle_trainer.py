"""
PaddleOCR Fine-Tuning - TÜBİTAK 3501 Projesi
=============================================
Deney #006: PaddleOCR Recognition Fine-Tuning (SVTR + CTC)

- Mimari: PP-OCRv4 Server Recognition (SVTR_LCNet)
- Decoder: CTC (karakter bazlı - tokenizer sorunu YOK)
- Türkçe dict.txt ile Türkçe karakterler tek sınıf
- Transfer learning: Pretrained PP-OCRv4 → Türkçe fine-tune

Kullanım:
    python src/paddle_trainer.py [epochs] [batch_size]
    python src/paddle_trainer.py 100 32
"""

import os
import sys
import subprocess
import shutil


# =============================================================================
# SABİTLER
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

PADDLEOCR_DIR = os.path.join(PROJECT_DIR, "PaddleOCR")
PRETRAINED_DIR = os.path.join(PADDLEOCR_DIR, "pretrained_models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "paddle-turkish-handwritten", "deney006")

DATA_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")
DICT_PATH = os.path.join(PROJECT_DIR, "data", "turkish_dict.txt")
TRAIN_LIST = os.path.join(DATA_DIR, "train_list.txt")
VAL_LIST = os.path.join(DATA_DIR, "val_list.txt")

# PP-OCRv4 Server Recognition config (derin mimari, Türkçe diyakritik için uygun)
BASE_CONFIG = "configs/rec/PP-OCRv4/PP-OCRv4_server_rec.yml"
PRETRAINED_MODEL_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_train.tar"


# =============================================================================
# KURULUM KONTROLLERI
# =============================================================================

def check_paddle():
    """PaddlePaddle kurulumunu doğrula."""
    try:
        import paddle
        print(f"[OK] PaddlePaddle {paddle.__version__}")
        print(f"     GPU: {paddle.device.is_compiled_with_cuda()}")
        if paddle.device.is_compiled_with_cuda():
            print(f"     CUDA: {paddle.device.cuda.device_count()} GPU bulundu")
        return True
    except ImportError:
        print("[HATA] PaddlePaddle yüklü değil!")
        print("       pip install paddlepaddle-gpu")
        return False


def check_paddleocr_repo():
    """PaddleOCR repo'sunun klonlanmış olduğunu doğrula."""
    if os.path.exists(os.path.join(PADDLEOCR_DIR, "tools", "train.py")):
        print(f"[OK] PaddleOCR repo: {PADDLEOCR_DIR}")
        return True
    else:
        print(f"[BİLGİ] PaddleOCR repo bulunamadı, klonlanıyor...")
        subprocess.run(
            ["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git", PADDLEOCR_DIR],
            check=True
        )
        print(f"[OK] PaddleOCR repo klonlandı: {PADDLEOCR_DIR}")
        return True


def download_pretrained_model():
    """Pretrained PP-OCRv4 server rec modelini indir."""
    model_dir = os.path.join(PRETRAINED_DIR, "ch_PP-OCRv4_rec_server_train")
    if os.path.exists(model_dir):
        print(f"[OK] Pretrained model mevcut: {model_dir}")
        return model_dir

    print(f"[BİLGİ] Pretrained model indiriliyor...")
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    tar_path = os.path.join(PRETRAINED_DIR, "ch_PP-OCRv4_rec_server_train.tar")

    # wget veya curl ile indir
    subprocess.run(
        ["python", "-c", f"""
import urllib.request
print('İndiriliyor... (bu biraz sürebilir)')
urllib.request.urlretrieve('{PRETRAINED_MODEL_URL}', '{tar_path.replace(os.sep, "/")}')
print('İndirme tamamlandı!')
"""],
        check=True
    )

    # Tar'ı aç
    import tarfile
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(PRETRAINED_DIR)
    os.remove(tar_path)

    print(f"[OK] Pretrained model indirildi: {model_dir}")
    return model_dir


def check_data():
    """Eğitim verilerinin hazır olduğunu doğrula."""
    errors = []

    if not os.path.exists(DICT_PATH):
        errors.append(f"dict.txt bulunamadı: {DICT_PATH}")

    if not os.path.exists(TRAIN_LIST):
        errors.append(f"train_list.txt bulunamadı: {TRAIN_LIST}")
        errors.append("Önce çalıştır: python src/convert_to_paddle.py")

    if not os.path.exists(VAL_LIST):
        errors.append(f"val_list.txt bulunamadı: {VAL_LIST}")

    if errors:
        for e in errors:
            print(f"[HATA] {e}")
        return False

    # Satır sayılarını kontrol et
    with open(TRAIN_LIST, 'r', encoding='utf-8') as f:
        train_count = sum(1 for _ in f)
    with open(VAL_LIST, 'r', encoding='utf-8') as f:
        val_count = sum(1 for _ in f)

    print(f"[OK] Veri: Train={train_count:,} | Val={val_count:,}")
    print(f"[OK] Dict: {DICT_PATH}")
    return True


# =============================================================================
# EĞİTİM
# =============================================================================

def train(epochs=100, batch_size=32):
    """PaddleOCR recognition modelini Türkçe için fine-tune et."""

    print("=" * 60)
    print("Deney #006: PaddleOCR Fine-Tuning")
    print("Mimari: PP-OCRv4 Server Recognition (SVTR)")
    print("Decoder: CTC (karakter bazlı)")
    print("=" * 60)

    # Kontroller
    if not check_paddle():
        return
    check_paddleocr_repo()
    pretrained_path = download_pretrained_model()
    if not check_data():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Özel Türkçe config dosyası (tüm path'ler relative, Windows uyumlu)
    config_path = os.path.join(PROJECT_DIR, "configs", "turkish_rec.yml")

    cmd = [
        "python", os.path.join(PADDLEOCR_DIR, "tools", "train.py"),
        "-c", config_path,
    ]

    print("\n" + "-" * 60)
    print("Eğitim başlıyor...")
    print(f"Config: {BASE_CONFIG}")
    print(f"Pretrained: {pretrained_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Çıktı: {OUTPUT_DIR}")
    print("-" * 60)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=PROJECT_DIR,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            print(f"\n[HATA] Eğitim hatası (exit code: {process.returncode})")
            return
    except KeyboardInterrupt:
        process.terminate()
        print("\n[Bilgi] Eğitim kullanıcı tarafından durduruldu.")

    print("\n" + "=" * 60)
    print(f"[OK] Eğitim tamamlandı! Model: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    train(epochs=epochs, batch_size=batch_size)
