"""
CSV → PaddleOCR Format Dönüştürücü
===================================
Mevcut data/synthetic/metadata.csv dosyasını PaddleOCR'ın beklediği
train_list.txt ve val_list.txt formatına dönüştürür.

PaddleOCR formatı (tab-separated):
    syn_000001.jpg\tMerkezi yağlama arızası
    syn_000002.jpg\t31390

Kullanım:
    python src/convert_to_paddle.py
"""

import os
import unicodedata
import pandas as pd


def convert_csv_to_paddle(
    csv_path,
    output_dir,
    train_ratio=0.8,
    random_state=42
):
    """
    metadata.csv → train_list.txt + val_list.txt

    Args:
        csv_path: metadata.csv yolu
        output_dir: Çıktı klasörü (train_list.txt ve val_list.txt yazılacak)
        train_ratio: Eğitim veri oranı (default: 0.8)
        random_state: Rastgelelik seed'i
    """
    print(f"[1/3] CSV okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Toplam veri: {len(df):,}")

    # Unicode NFC normalization
    df['text'] = df['text'].astype(str).apply(lambda x: unicodedata.normalize('NFC', x))

    # Train/Val split
    print(f"\n[2/3] Train/Val ayrılıyor ({train_ratio:.0%} / {1-train_ratio:.0%})...")
    train_df = df.sample(frac=train_ratio, random_state=random_state)
    val_df = df.drop(train_df.index)
    print(f"      Eğitim: {len(train_df):,} | Validation: {len(val_df):,}")

    # PaddleOCR formatında yaz
    print(f"\n[3/3] PaddleOCR formatında yazılıyor...")

    train_path = os.path.join(output_dir, "train_list.txt")
    val_path = os.path.join(output_dir, "val_list.txt")

    _write_paddle_list(train_df, train_path)
    _write_paddle_list(val_df, val_path)

    print(f"      [OK] {train_path}")
    print(f"      [OK] {val_path}")

    # Doğrulama
    print(f"\n[Doğrulama] İlk 5 satır (train_list.txt):")
    with open(train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"      {line.strip()}")

    return train_path, val_path


def _write_paddle_list(df, output_path):
    """DataFrame'i PaddleOCR label formatında yazar."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            file_name = row['file_name']
            text = str(row['text']).strip()
            # Tab-separated: dosya_adı\tmetin
            f.write(f"{file_name}\t{text}\n")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    CSV_PATH = os.path.join(PROJECT_DIR, "data", "synthetic", "metadata.csv")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")

    convert_csv_to_paddle(CSV_PATH, OUTPUT_DIR)
