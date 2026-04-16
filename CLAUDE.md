# FormReader - Proje Rehberi

> Bu dosya projenin hafızasıdır. Her oturumda önce bunu oku.
> Teknik detaylar için: `docs/3501_basvuru_formu.pdf`

---

## Proje Nedir?

**FormReader** - TÜBİTAK 3501 Kariyer Programı kapsamında geliştirilen, üretim tesislerindeki el yazısı formlardan otomatik veri çıkarma sistemi.

### Amaç
Üretim takip formlarındaki:
- Duruş süreleri (dakika)
- Duruş sebepleri (el yazısı Türkçe metin)
- Operatör isimleri
- Tarih/saat bilgileri
- Üretim miktarları

...gibi verileri OCR ile okuyup, NLP ile kategorize etmek.

### Hedef Doğruluk Oranları
| Alan | Yöntem | Hedef |
|------|--------|-------|
| Sayısal veriler (süre, miktar) | TrOCR + Tesseract | **%99** |
| El yazısı Türkçe metin | TrOCR fine-tuned | **%85** |
| Duruş sebebi gruplama | NLP (spaCy/BERT) | **%95** |

---

## MEVCUT MİMARİ (Mart 2025 - Deney #003)

### TrOCR Encoder + BERTurk Decoder

```
┌─────────────────────────────────────────────────────────┐
│              TrOCR Türkçe Mimarisi (Deney #003)         │
├─────────────────────────────────────────────────────────┤
│  ENCODER                    DECODER                     │
│  ────────                   ────────                    │
│  TrOCR (OCR-trained)        BERTurk (Türkçe BERT)      │
│  microsoft/trocr-base       dbmdz/bert-base-turkish    │
│  -handwritten               -cased                      │
│                                                         │
│  Görüntü → Metin Feature →  → Türkçe Token'lar → Metin │
│  (OCR için eğitilmiş)       (Türkçe optimize)          │
└─────────────────────────────────────────────────────────┘
```

### Neden Bu Mimari?
- **Deney #001** (%13): TrOCR encoder (iyi göz) + RoBERTa decoder (kötü dil)
- **Deney #002** (%4): Generic ViT (kötü göz) + BERTurk decoder (iyi dil)
- **Deney #003**: TrOCR encoder (iyi göz) + BERTurk decoder (iyi dil) = İkisinin en iyisi

### Bu Hala TrOCR mı?
**EVET!** TrOCR bir mimari pattern'dir (Vision Encoder + Text Decoder).
Paper'da açıkça belirtilmiş:
> "TrOCR can be easily extended to multilingual text recognition by
> leveraging multilingual pre-trained models on the decoder side."

---

## MİMARİ YAKLAŞIM

### Tercih Edilen: TrOCR
- TÜBİTAK başvuru formunda **TrOCR** belirtilmiştir
- Mevcut mimari: TrOCR Encoder + BERTurk Decoder
- Beklenen başarıya ulaşılamazsa alternatifler (EasyOCR, PaddleOCR) değerlendirilebilir
- Yöntem değişikliği gerekçeyle `experiments/future_plans.md`'ye kaydedilmeli

### Tesseract (Hibrit)
- **Sayısal veriler** için yardımcı olarak kullanılabilir
- Ana OCR motoru değil, destek amaçlı

### Web Araştırma Kuralı
- Problemlerde **web search** yapılmalı (GitHub, HuggingFace, arxiv, Stack Overflow)
- Sonuçlar `docs/web_search.md`'ye kaydedilmeli

### Bağımlılık Yönetimi Kuralı (KRİTİK)
Her yeni paket kurulduğunda veya versiyon değişikliği yapıldığında `requirements.txt` GÜNCELLENMELİ.

**Kurallar:**
1. Yeni paket kurulunca → emin olduktan sonra (çalıştığı doğrulandıktan sonra) `requirements.txt`'ye **versiyon kilidi ile** eklenir (`paket==X.Y.Z`)
2. Her pakete yorum satırı eklenir: hangi modül/deney için gerekli
3. Versiyon değişikliği yapılırsa (downgrade/upgrade) → yorum olarak NEDEN'i yazılır
4. Paket silmeden önce → proje içinde kullanımı grep'lenir, başka yerde kullanılmıyorsa silinir
5. Yeni paket **mevcut paketleri kırmamalı** (`pip check` ile doğrula)
6. Windows-özel kurulum komutları (CUDA wheel vs.) `requirements.txt` başında belirtilir

**Neden önemli:**
- Yeni bir makinede projeyi kurmak için tek referans `requirements.txt`
- TrOCR → PaddleOCR geçişinde numpy/torch/paddle uyumsuzlukları saatler kaybettirdi
- TÜBİTAK teslim aşamasında jüriye veriyoruz, çalışır olmalı

**Her deney sonrası kontrol:**
- Yeni paket kuruldu mu? → requirements.txt'ye ekle
- Versiyon değişti mi? → requirements.txt'yi güncelle
- Commit et

### Detaylı Bilgi
Teknik gereksinimler ve proje detayları için:
```
docs/3501_basvuru_formu.pdf    # TÜBİTAK başvuru formu
docs/coding_rules.md           # Kodlama kuralları ve standartlar
docs/literature_review.md      # Literatür taraması (15 makale analizi)
docs/web_search.md             # Web araştırma kayıtları
```
Bu dosyaları oku, projenin kapsamını, kısıtlamalarını ve referanslarını anla.

---

## Dosya Yapısı

```
FormReader/
│
├── src/                          # KAYNAK KODLAR
│   ├── ocr_engine.py            # TrOCR inference motoru
│   ├── trainer.py               # Model eğitim scripti
│   ├── data_generator.py        # Sentetik veri üretici
│   └── FormReader_Main.ipynb    # Ana çalışma notebook'u
│
├── data/                         # VERİLER (gitignore'da)
│   ├── fonts/                   # El yazısı fontları (.ttf)
│   │   └── *.ttf                # Minimum 5-10 farklı font
│   ├── synthetic/               # Üretilen sentetik görüntüler
│   │   ├── syn_000000.jpg       # Görüntü dosyaları
│   │   └── metadata.csv         # Etiketler (file_name, text, category)
│   └── real_forms/              # Gerçek form görüntüleri (test için)
│
├── models/                       # EĞİTİLMİŞ MODELLER (gitignore'da)
│   └── trocr-turkish-handwritten/
│       ├── deney003/            # Her deney AYRI klasöre kaydedilir
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── preprocessor_config.json
│       │   └── tokenizer files...
│       ├── deney004/            # Gelecek deneyler
│       └── ...
│
├── experiments/                  # DENEY KAYITLARI (gitignore'da)
│   ├── experiment_log.md        # Tamamlanan deney sonuçları
│   └── future_plans.md          # Planlanan deneyler, gerekçeler, referanslar
│
├── docs/                         # DÖKÜMANLAR (gitignore'da)
│   ├── 3501_basvuru_formu.pdf   # TÜBİTAK başvuru formu
│   ├── literature_review.md     # Literatür taraması ve makale analizi
│   ├── coding_rules.md          # Kodlama kuralları ve standartlar
│   ├── web_search.md            # Web araştırma kayıtları
│   └── cites/                   # Referans makaleler (15 PDF)
│
├── CLAUDE.md                     # BU DOSYA - Proje rehberi
├── .gitignore                    # Git ignore kuralları
└── requirements.txt              # Python bağımlılıkları
```

---

## Kaynak Kod Detayları

### 1. ocr_engine.py
**Amaç:** Eğitilmiş TrOCR modeliyle görüntüden metin çıkarma

```python
from src.ocr_engine import TrOCREngine

# Otomatik olarak fine-tuned model varsa onu, yoksa base model yükler
engine = TrOCREngine()

# Tek görüntü
result = engine.predict("image.jpg")
print(result)  # "Merkezi yağlama arızası"

# Batch işlem
results = engine.predict_batch(["img1.jpg", "img2.jpg"])
```

**Önemli Parametreler (generate):**
```python
generated_ids = model.generate(
    pixel_values,
    max_length=64,          # Maksimum çıktı uzunluğu
    num_beams=4,            # Beam search genişliği
    early_stopping=True,
    length_penalty=2.0,
    # NOT: no_repeat_ngram_size KULLANMA - sayıları kesiyor!
)
```

### 2. trainer.py
**Amaç:** TrOCR modelini Türkçe el yazısı için fine-tune etme

**Mimari:** TrOCR Encoder (OCR-trained) + BERTurk Decoder (Türkçe)

```python
from src.trainer import train_trocr

train_trocr(
    dataset_dir="data/synthetic",
    csv_path="data/synthetic/metadata.csv",
    output_model_dir="models/trocr-turkish-handwritten",
    epochs=15,              # Epoch sayısı (deney bazında ayarlanabilir)
    batch_size=8,           # GPU'ya göre ayarla (4-16)
    learning_rate=2e-5,     # Önerilen: 2e-5 ile 5e-5 arası
    resume_from_checkpoint=None  # Checkpoint'tan devam için path
)
```

**Model Bileşenleri (trainer.py içinde tanımlı):**
```python
BASE_MODEL = "microsoft/trocr-base-handwritten"   # TrOCR (encoder korunur)
DECODER_MODEL = "dbmdz/bert-base-turkish-cased"    # Türkçe decoder (BERTurk)
```

**Windows/CUDA Kritik Ayarları:**
```python
# trainer.py içinde bu ayarlar VAR ve DEĞİŞTİRME:
dataloader_num_workers=0      # Windows multiprocessing hatası önler
predict_with_generate=False   # VRAM taşmasını önler
gradient_checkpointing=True   # VRAM tasarrufu sağlar
eval_accumulation_steps=8     # Eval sırasında bellek yönetimi
```

### 3. data_generator.py
**Amaç:** Sentetik el yazısı görüntüleri üretme

```python
from src.data_generator import generate_dataset

csv_path = generate_dataset(
    font_dir="data/fonts",
    output_dir="data/synthetic",
    count=10000  # Üretilecek görüntü sayısı
)
```

**Veri Dağılımı (Otomatik):**
- %30 Sayılar (0, 140, 1300, 687775, %85, 0,5 vb.)
- %30 Türkçe Metinler (isimler, duruş sebepleri, ürün adları)
- %20 Tarihler (06.03.2023, 08:30, Salı vb.)
- %15 Karışık (Hat-3, 10 LT, A1 Arıza vb.)
- %5 Teknik Kodlar (S25C-925103023031 vb.)

**Augmentation (Görüntü Zenginleştirme):**
- Döndürme (±3.5°)
- Gaussian gürültü
- Bulanıklık (0.3-0.9)
- Kontrast/parlaklık değişimi
- Farklı mürekkep renkleri (siyah, mavi, koyu mavi)
- Farklı kağıt tonları (beyaz, gri, sarımsı)

---

## Bilinen Sorunlar ve Çözümler

### SORUN 1: Türkçe Karakterler Bozuk Çıkıyor
**Belirti:** "Üretim" → "?retim" veya "retim"

**Neden:** TrOCR'ın decoder'ı RoBERTa tokenizer kullanıyor. Bu tokenizer İngilizce için eğitilmiş. Türkçe karakterler (ı, ğ, ü, ş, ö, ç, İ, Ğ, Ü, Ş, Ö, Ç) byte-level BPE olarak çoklu token'lara ayrılıyor. Model bu token dizilerini doğru üretmeyi öğrenemiyor.

**Denenen Çözümler:**
- 10.000 veri ile 10 epoch: Başarısız (%13)

**Denenecek Çözümler:**
- Daha fazla veri (50K-100K)
- Daha fazla epoch (20-30)
- Türkçe ağırlıklı veri seti (%70 Türkçe)
- Multilingual TrOCR modeli araştırması

### SORUN 2: Sayılar Kesiliyor (100200 → 1002)
**Belirti:** Uzun sayılar eksik çıkıyor

**Neden:** `no_repeat_ngram_size=3` parametresi "00" veya "11" gibi tekrarları engelliyor.

**Çözüm:** ocr_engine.py'den `no_repeat_ngram_size` parametresi KALDIRILDI. ✓

### SORUN 3: GPU Bulunamadı / 119 Saat Tahmini
**Belirti:** Jupyter'da eğitim CPU'da çalışıyor

**Neden:** PyTorch CPU versiyonu yüklü veya kernel restart gerekiyor

**Çözüm:**
```python
# Kontrol et
import torch
print(torch.cuda.is_available())  # True olmalı
print(torch.version.cuda)         # 11.8 veya 12.x olmalı

# False ise: PyTorch'u CUDA ile yeniden yükle
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### SORUN 4: PC Çöküyor / VRAM Taşması
**Belirti:** Eğitim sırasında sistem donuyor

**Neden:** `predict_with_generate=True` evaluation sırasında çok fazla VRAM kullanıyor

**Çözüm:** trainer.py'de şu ayarlar VAR:
```python
predict_with_generate=False
gradient_checkpointing=True
eval_accumulation_steps=8
```

---

## Deney Kayıtları

**ÖNEMLİ:** Yeni deneme yapmadan önce mutlaka her iki dosyayı da oku:

### 1. Tamamlanan Deneyler
```
experiments/experiment_log.md
```
- Hangi parametrelerle ne denendi
- Sonuçlar ne oldu
- Neden başarısız/başarılı oldu
- Çıkarımlar ve karşılaştırmalı analiz

### 2. Planlanan Deneyler
```
experiments/future_plans.md
```
- Sıradaki deneylerin gerekçeleri ve teknik yaklaşımları
- Referans paper'lar ve kaynakçalar
- Karar ağacı (hangi sonuca göre hangi deneye geçilecek)
- Tamamlanan planların sonuçları

**Aynı hatayı iki kez yapma, önce kayıtları kontrol et!**

---

## Hızlı Başlangıç

### Yeni Veri Üretmek
```python
# Jupyter'da veya Python'da
import sys
sys.path.append("c:/Users/User/Desktop/FormReader")

from src.data_generator import generate_dataset

generate_dataset(
    font_dir="c:/Users/User/Desktop/FormReader/data/fonts",
    output_dir="c:/Users/User/Desktop/FormReader/data/synthetic",
    count=10000
)
```

### Model Eğitmek
```python
from src.trainer import train_trocr

train_trocr(
    dataset_dir="c:/Users/User/Desktop/FormReader/data/synthetic",
    csv_path="c:/Users/User/Desktop/FormReader/data/synthetic/metadata.csv",
    output_model_dir="c:/Users/User/Desktop/FormReader/models/trocr-turkish-handwritten",
    epochs=10,
    batch_size=8,
    learning_rate=4e-5
)
```

### Test Etmek
```python
from src.ocr_engine import TrOCREngine

engine = TrOCREngine()

# Tek görüntü test
result = engine.predict("c:/Users/User/Desktop/FormReader/data/synthetic/syn_003000.jpg")
print(f"Tahmin: {result}")
```

---

## Notlar

### Eğitim Süresi Tahmini
- GPU (RTX serisi): Epoch başı ~40 dakika
- 10 epoch: ~6-7 saat
- 20 epoch: ~13-14 saat

### Önerilen Eğitim Parametreleri
| Parametre | Başlangıç | Alternatif |
|-----------|-----------|------------|
| epochs | 10 | 20-30 |
| batch_size | 8 | 4 (düşük VRAM), 16 (yüksek VRAM) |
| learning_rate | 4e-5 | 2e-5 (daha stabil), 5e-5 (daha agresif) |
| warmup_steps | 500 | veri sayısının %5-10'u |

### Base Model
- **microsoft/trocr-base-handwritten**
- İngilizce el yazısı için eğitilmiş (IAM dataset)
- Encoder: ViT (Vision Transformer)
- Decoder: RoBERTa
- Boyut: ~334M parametre

---

## İletişim ve Kaynaklar

- TÜBİTAK Proje Detayları: `docs/3501_basvuru_formu.pdf`
- Kodlama Kuralları: `docs/coding_rules.md`
- Literatür Taraması: `docs/literature_review.md`
- Referans Makaleler: `docs/cites/` (15 PDF)
- Deney Geçmişi: `experiments/experiment_log.md`
- Gelecek Planlar: `experiments/future_plans.md`
- TrOCR Paper: https://arxiv.org/abs/2109.10282
- TrOCR İspanyolca Paper: https://arxiv.org/abs/2407.06950
- HuggingFace Model: https://huggingface.co/microsoft/trocr-base-handwritten
- BERTurk Model: https://huggingface.co/dbmdz/bert-base-turkish-cased
