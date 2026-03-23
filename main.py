import os
import sys
# src klasörünü dahil et
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import find_boxes
from utils import parse_xml_labels, match_and_crop
from ocr_engine import OCREngine

def main():
    # --- AYARLAR ---
    IMAGE_PATH = "data/raw/S25C-925103023031.jpg" # Senin resmin
    XML_PATH = "data/xml_labels/S25C-925103023031.xml"   # Senin XML'in
    
    print("--- Form Reader Başlatılıyor ---")

    # 1. Görüntü İşleme (Kutuları Bul)
    print("1. Form taranıyor ve kutular bulunuyor...")
    processed_image, boxes = find_boxes(IMAGE_PATH, debug=True)

    # 2. Etiketleri Eşleştir ve Kes
    print("2. Etiketler XML'den okunuyor ve veriler kesiliyor...")
    xml_labels = parse_xml_labels(XML_PATH)
    labeled_data = match_and_crop(processed_image, boxes, xml_labels)
    
    print(f"   -> Toplam {len(labeled_data)} adet etiketli veri kesildi.")

    # 3. OCR ile Okuma (Modeli Başlat)
    ocr = OCREngine()
    
    print("\n--- OCR OKUMA SONUÇLARI ---")
    print(f"{'ETİKET':<20} | {'OKUNAN DEĞER'}")
    print("-" * 40)

    for item in labeled_data:
        # Kesilen küçük kutucuğu modele ver
        text = ocr.predict(item['roi_image'])
        print(f"{item['label']:<20} | {text}")

if __name__ == "__main__":
    main()