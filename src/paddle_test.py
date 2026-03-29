"""
PaddleOCR Test - TÜBİTAK 3501 Projesi
======================================
Eğitilmiş PaddleOCR modelini sentetik veri üzerinde test eder.

Kullanım:
    python src/paddle_test.py [model_dir]
"""

import os
import sys
import unicodedata
import pandas as pd
from PIL import Image


def test_paddle_model(
    model_dir=None,
    data_dir=None,
    csv_path=None,
    samples_per_category=5,
    random_state=42
):
    """
    PaddleOCR modelini sentetik veri ile test eder.

    Args:
        model_dir: Eğitilmiş model dizini (None ise son deney kullanılır)
        data_dir: Sentetik veri dizini
        csv_path: metadata.csv yolu
        samples_per_category: Her kategoriden kaç örnek test edilecek
        random_state: Rastgelelik seed'i
    """
    from paddleocr import PaddleOCR

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

    # PaddleOCR yükle (sadece recognition, detection yok)
    ocr = PaddleOCR(
        rec_model_dir=model_dir,
        rec_char_dict_path=dict_path,
        use_space_char=True,
        use_angle_cls=False,
        use_gpu=True,
    )

    # Veri yükle
    df = pd.read_csv(csv_path)
    categories = ['number', 'turkish_text', 'date', 'mixed', 'code']

    correct = 0
    total = 0

    print("\n" + "=" * 60)
    print("SENTETİK VERİ TESTİ (PaddleOCR)")
    print("=" * 60)

    for cat in categories:
        print(f"\n--- {cat.upper()} ---")
        cat_df = df[df['category'] == cat]
        if len(cat_df) == 0:
            continue
        samples = cat_df.sample(min(samples_per_category, len(cat_df)), random_state=random_state)

        for _, row in samples.iterrows():
            img_path = os.path.join(data_dir, row['file_name'])
            expected = unicodedata.normalize('NFC', str(row['text']).strip())

            # PaddleOCR ile tahmin
            result = ocr.ocr(img_path, det=False, cls=False)

            if result and result[0]:
                predicted = result[0][0][0]  # İlk sonucun metni
                confidence = result[0][0][1]
            else:
                predicted = ""
                confidence = 0.0

            predicted = predicted.strip()

            match = "✓" if predicted == expected else "✗"
            if match == "✓":
                correct += 1
            total += 1

            print(f"{match} Beklenen: {expected:<25} | Tahmin: {predicted:<25} | Güven: {confidence:.2f}")

    print("\n" + "=" * 60)
    print(f"DOĞRULUK: {correct}/{total} = {100*correct/total:.1f}%")
    print("=" * 60)

    return correct, total


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    test_paddle_model(model_dir=model_dir)
