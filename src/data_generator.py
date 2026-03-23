"""
Sentetik Veri Üretici - TÜBİTAK 3501 Projesi
============================================
Hedef: %99 sayısal, %85 metin doğruluğu için optimize edilmiş veri seti

Dağılım (10.000 veri):
- %30 Sayılar (3.000)     → Kritik: %99 hedefi
- %30 Türkçe Metinler (3.000)
- %20 Tarihler (2.000)
- %15 Karışık (1.500)
- %5  Teknik Kodlar (500)
"""

import os
import random
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime, timedelta


# =============================================================================
# TÜRKÇE VERİ HAVUZLARI
# =============================================================================

# Türkçe karakterli isimler (ı, ğ, ü, ş, ö, ç içerenler öncelikli)
TURKCE_ISIMLER = [
    # Erkek isimleri (Türkçe karakterli)
    "Işık", "Işıl", "İbrahim", "İsmail", "İlhan", "İlker", "İzzet",
    "Şükrü", "Şenol", "Şahin", "Şeref", "Şevket",
    "Müslüm", "Mümin", "Münir", "Müfit",
    "Güneş", "Gürol", "Gürkan", "Güven", "Gökhan", "Göksel",
    "Ömer", "Özgür", "Özkan", "Özen", "Özdemir",
    "Çetin", "Çağlar", "Çağdaş", "Çağrı",
    "Ağa", "Doğan", "Erdoğan", "Tuğrul",
    # Kadın isimleri (Türkçe karakterli)
    "Ayşe", "Ayşegül", "Ayşenur", "Işıl", "İpek", "İlknur",
    "Şule", "Şeyma", "Şerife", "Şirin",
    "Müge", "Müjde", "Münevver",
    "Gülşen", "Gülnur", "Gülbahar", "Gülay", "Gözde",
    "Özlem", "Özge", "Öznur",
    "Çiğdem", "Çağla",
    "Tuğba", "Tuğçe", "Doğa",
    # Normal isimler
    "Ahmet", "Mehmet", "Ali", "Mustafa", "Hasan", "Hüseyin",
    "Fatma", "Zeynep", "Elif", "Merve", "Esra", "Büşra",
]

TURKCE_SOYADLAR = [
    # Türkçe karakterli soyadlar
    "Yılmaz", "Yıldız", "Yıldırım", "Işık", "Işıklı",
    "Şahin", "Şen", "Şener", "Şimşek", "Şeker",
    "Gül", "Güneş", "Güler", "Gündüz", "Gök", "Göker",
    "Öztürk", "Özdemir", "Özkan", "Özen", "Özçelik",
    "Çelik", "Çetin", "Çınar", "Çakır", "Çolak",
    "Doğan", "Doğru", "Tuğcu", "Ağaoğlu",
    "Ünal", "Ünlü", "Üstün", "Üçer",
    # Normal soyadlar
    "Kaya", "Demir", "Arslan", "Aydın", "Koç", "Kurt",
    "Erdoğan", "Polat", "Aksoy", "Taş", "Yavuz", "Bulut",
]

# Gerçek üretim duruş sebepleri (formdan alındı)
DURUS_SEBEPLERI = [
    # Mekanik arızalar
    "Merkezi yağlama arıza", "Merkezi yağlama arızası",
    "Bant kopması", "Bant koptu", "2 nolu şiling geçiş bant kopması",
    "Konveyör arızası", "Konveyör bant kopması", "Konveyör durdu",
    "Motor arızası", "Motor yandı", "Motor aşırı ısındı",
    "Rulman arızası", "Rulman değişimi", "Yatak arızası",
    "Hidrolik arıza", "Hidrolik kaçağı", "Hidrolik basınç düştü",
    "Pnömatik arıza", "Hava basıncı düştü", "Kompresör arızası",

    # Elektrik arızaları
    "Elektrik kesintisi", "Elektrik arızası", "Sigorta attı",
    "Sensör arızası", "Sensör bozuldu", "Fotosel arızası",
    "PLC hatası", "Yazılım hatası", "Sistem kilitlendi",
    "Kablo koptu", "Bağlantı gevşedi", "Kontaktör arızası",

    # Üretim duruşları
    "Ürün değişimi", "Kalıp değişimi", "Format değişimi",
    "Baskıval ve riban değişmesi", "Etiket değişimi",
    "Hammadde bekleme", "Malzeme bekleme", "Preform bekleme",
    "Kapak bekleme", "Etiket bekleme", "Shrink bekleme",

    # Temizlik ve bakım
    "Planlı bakım", "Periyodik bakım", "Arızi bakım",
    "Temizlik", "Makine temizliği", "Hat temizliği",
    "Otomatik geçiş ve tambur temizlik",
    "Yağlama", "Yağ değişimi", "Filtre değişimi",

    # Tank ve sistem arızaları
    "Ozon tankı arızası", "Ozon tankı taştı", "Ozon tankı",
    "Su bekleme", "Su arızası", "Şebeke suyu kesildi",
    "Buhar arızası", "Kazan arızası", "Isıtma arızası",

    # Operatör kaynaklı
    "Operatör hatası", "Yanlış ayar", "Hatalı besleme",
    "Mola", "Yemek molası", "Vardiya değişimi",
    "Eğitim", "Toplantı", "Ziyaretçi",

    # Diğer
    "Üretim başlama", "Üretim başlangıcı", "Üretim sonu",
    "Forklift bekleme", "Palet bekleme", "Sevkiyat bekleme",
    "Kalite kontrol", "Numune alma", "Test üretimi",
]

# Ürün tanımları
URUN_TANIMLARI = [
    "0,5 LT", "0.5 LT", "0,5 Lt", "0.5 Lt",
    "1 LT", "1 Lt", "1,5 LT", "1.5 LT", "1,5 Lt",
    "2 LT", "2,5 LT", "5 LT", "10 LT", "19 LT",
    "Pet 3", "PET 3", "Pet-3", "PET-3",
    "Şişe", "Damacana", "Bidon",
    "Fıratköy", "Erikli", "Hayat", "Pınar",
]

# Günler (Türkçe)
GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
GUNLER_KISA = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


# =============================================================================
# VERİ ÜRETİCİLER
# =============================================================================

def generate_number():
    """
    Sayısal veri üretici - %99 hedefi için kritik!
    Üretim formlarındaki gerçek sayı formatları
    """
    choice = random.choice([
        'tam_kucuk',      # 0-999
        'tam_orta',       # 1000-99999
        'tam_buyuk',      # 100000-999999
        'ondalik_virgul', # 0,5 / 1,5
        'ondalik_nokta',  # 0.5 / 1.5
        'sure_dk',        # 05, 10, 130
        'sure_saat',      # 600, 1800
        'yuzde',          # %85, %99
        'miktar',         # 1300, 2200
    ])

    if choice == 'tam_kucuk':
        return str(random.randint(0, 999))
    elif choice == 'tam_orta':
        return str(random.randint(1000, 99999))
    elif choice == 'tam_buyuk':
        return str(random.randint(100000, 999999))
    elif choice == 'ondalik_virgul':
        return f"{random.randint(0, 99)},{random.randint(0, 9)}"
    elif choice == 'ondalik_nokta':
        return f"{random.randint(0, 99)}.{random.randint(0, 9)}"
    elif choice == 'sure_dk':
        # Dakika süreleri (bazen başında 0 ile)
        val = random.randint(1, 180)
        return f"{val:02d}" if random.random() > 0.5 else str(val)
    elif choice == 'sure_saat':
        return str(random.choice([480, 540, 600, 720, 960, 1200, 1440, 1800]))
    elif choice == 'yuzde':
        return f"%{random.randint(0, 100)}"
    else:  # miktar
        return str(random.choice([140, 1300, 2200, 2400, 29400, 352800]))


def generate_turkish_text():
    """
    Türkçe metin üretici - Türkçe karakterlere özel vurgu
    """
    choice = random.choice([
        'isim_soyisim',
        'durus_sebebi',
        'urun_tanimi',
        'aciklama',
        'tek_kelime',
    ])

    if choice == 'isim_soyisim':
        isim = random.choice(TURKCE_ISIMLER)
        soyad = random.choice(TURKCE_SOYADLAR)
        # Farklı formatlar
        fmt = random.choice([
            f"{isim} {soyad}",
            f"{isim.upper()} {soyad.upper()}",
            f"{isim} {soyad.upper()}",
        ])
        return fmt

    elif choice == 'durus_sebebi':
        return random.choice(DURUS_SEBEPLERI)

    elif choice == 'urun_tanimi':
        return random.choice(URUN_TANIMLARI)

    elif choice == 'aciklama':
        # Kısa açıklamalar
        templates = [
            f"Hat {random.randint(1,4)} durdu",
            f"Makine {random.randint(1,10)} arızalı",
            f"{random.choice(GUNLER)} günü bakım",
            f"Vardiya {random.randint(1,3)} raporu",
            f"{random.choice(['Şişirme', 'Dolum', 'Etiket', 'Paketleme'])} hattı",
        ]
        return random.choice(templates)

    else:  # tek_kelime - Türkçe karakterli kelimeler
        turkce_kelimeler = [
            # ı içerenler
            "arıza", "bakım", "ışık", "sıcaklık", "basınç", "akış",
            "kırık", "sızıntı", "tıkanık", "yapışık", "kayış",
            # ğ içerenler
            "yağlama", "soğutma", "bağlantı", "değişim", "düğme",
            "sağlam", "boğaz", "dağıtım", "çağrı",
            # ü içerenler
            "üretim", "gürültü", "süre", "ürün", "büyük", "küçük",
            "düşük", "yüksek", "süzgeç", "düzeltme",
            # ş içerenler
            "başlangıç", "geçiş", "döküş", "taşma", "şişirme",
            "işlem", "döşeme", "taşıma", "başarılı",
            # ö içerenler
            "ölçüm", "dönem", "gösterge", "önemli", "kontrol",
            "sönüm", "dönüş", "bölüm", "görev",
            # ç içerenler
            "çalışma", "geçici", "açıklama", "içerik", "dışarı",
            "çıkış", "geçit", "uçuş", "seçim",
            # İ içerenler (büyük)
            "İşlem", "İnceleme", "İzleme", "İyileştirme",
        ]
        return random.choice(turkce_kelimeler)


def generate_date():
    """
    Tarih ve zaman üretici - Formda sık kullanılan formatlar
    """
    # Rastgele tarih (2020-2025 arası)
    start = datetime(2020, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    random_date = start + timedelta(days=random.randint(0, delta.days))

    choice = random.choice([
        'tarih_noktali',    # 06.03.2023
        'tarih_tireli',     # 06-03-2023
        'tarih_slash',      # 06/03/2023
        'tarih_kisa',       # 06.03.23
        'vardiya',          # 08-18 veya 08:00-18:00
        'saat',             # 08:30, 14:45
        'gun_tarih',        # Salı 06.03
    ])

    d, m, y = random_date.day, random_date.month, random_date.year

    if choice == 'tarih_noktali':
        return f"{d:02d}.{m:02d}.{y}"
    elif choice == 'tarih_tireli':
        return f"{d:02d}-{m:02d}-{y}"
    elif choice == 'tarih_slash':
        return f"{d:02d}/{m:02d}/{y}"
    elif choice == 'tarih_kisa':
        return f"{d:02d}.{m:02d}.{str(y)[2:]}"
    elif choice == 'vardiya':
        baslangic = random.choice([6, 7, 8, 14, 22])
        bitis = baslangic + random.choice([8, 10, 12])
        if random.random() > 0.5:
            return f"{baslangic:02d}-{bitis:02d}"
        else:
            return f"{baslangic:02d}:00-{bitis:02d}:00"
    elif choice == 'saat':
        saat = random.randint(0, 23)
        dakika = random.choice([0, 15, 30, 45])
        return f"{saat:02d}:{dakika:02d}"
    else:  # gun_tarih
        gun = random.choice(GUNLER_KISA)
        return f"{gun} {d:02d}.{m:02d}"


def generate_mixed():
    """
    Karışık veri üretici - Sayı + metin kombinasyonları
    """
    choice = random.choice([
        'birim_sayi',
        'hat_numara',
        'kod_aciklama',
        'miktar_birim',
    ])

    if choice == 'birim_sayi':
        birimler = ["LT", "Lt", "kg", "Kg", "KG", "adet", "Adet", "paket", "palet"]
        return f"{random.randint(1, 100)} {random.choice(birimler)}"

    elif choice == 'hat_numara':
        prefixes = ["Hat", "HAT", "Makine", "Vardiya", "Operasyon", "İstasyon"]
        separators = ["-", " ", ""]
        return f"{random.choice(prefixes)}{random.choice(separators)}{random.randint(1, 10)}"

    elif choice == 'kod_aciklama':
        kodlar = ["A1", "B2", "C3", "K1", "K2", "M1", "M2", "E1", "E2"]
        aciklamalar = ["Arıza", "Bakım", "Değişim", "Temizlik", "Ayar"]
        return f"{random.choice(kodlar)} {random.choice(aciklamalar)}"

    else:  # miktar_birim
        return f"{random.randint(100, 10000)} {random.choice(['adet', 'kg', 'lt'])}"


def generate_technical_code():
    """
    Teknik kod üretici - Form numaraları, seri kodları
    """
    choice = random.choice([
        'form_kodu',
        'seri_no',
        'barkod',
    ])

    if choice == 'form_kodu':
        # S25C-925103023031 formatı
        prefix = random.choice(["S25C", "F01", "FR", "KR", "PT"])
        mid = "".join([str(random.randint(0, 9)) for _ in range(random.randint(8, 12))])
        sep = random.choice(["-", "", " "])
        return f"{prefix}{sep}{mid}"

    elif choice == 'seri_no':
        return "".join([str(random.randint(0, 9)) for _ in range(random.randint(6, 10))])

    else:  # barkod
        return "".join([str(random.randint(0, 9)) for _ in range(13)])


# =============================================================================
# GÖRÜNTÜ ÜRETİCİ
# =============================================================================

def create_synthetic_image(text, font_path, output_path):
    """
    Sentetik el yazısı görüntüsü üretir.
    Form kağıdı ve mürekkep simülasyonu ile gerçekçi sonuçlar.
    """
    # Unicode NFC Normalization - Türkçe karakter tutarlılığı
    text = unicodedata.normalize('NFC', text)
    # 1. Font Ayarları
    font_size = random.randint(28, 52)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"[Hata] Font yüklenemedi: {font_path}")
        return False

    # 2. Metin Boyutu Hesaplama
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Padding (kenar boşluğu)
    pad_x = random.randint(20, 40)
    pad_y = random.randint(15, 30)
    W = text_w + pad_x * 2
    H = text_h + pad_y * 2

    # 3. Arka Plan - Form kağıdı simülasyonu
    # Hafif gri tonları (taranmış kağıt efekti)
    bg_val = random.randint(235, 255)
    # Bazen hafif sarımsı (eski kağıt)
    if random.random() > 0.8:
        bg_color = (bg_val, bg_val - random.randint(0, 10), bg_val - random.randint(5, 15))
    else:
        bg_color = (bg_val, bg_val, bg_val)

    image = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(image)

    # 4. Yazı Rengi - Mürekkep simülasyonu
    # Çoğunlukla koyu mavi/siyah (tükenmez kalem)
    ink_choice = random.choice(['siyah', 'mavi', 'koyu_mavi'])
    if ink_choice == 'siyah':
        r = random.randint(10, 50)
        ink_color = (r, r, r)
    elif ink_choice == 'mavi':
        ink_color = (random.randint(0, 30), random.randint(0, 50), random.randint(80, 150))
    else:  # koyu_mavi
        ink_color = (random.randint(0, 20), random.randint(0, 30), random.randint(50, 100))

    # Metni yaz (hafif rastgele offset ile)
    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-2, 2)
    x = (W - text_w) // 2 + offset_x
    y = (H - text_h) // 2 + offset_y
    draw.text((x, y), text, font=font, fill=ink_color)

    # 5. AUGMENTATION (Veri Çoğaltma / Zorlaştırma)

    # 5.1 Döndürme (el yazısı hiçbir zaman düz değildir)
    if random.random() > 0.2:
        angle = random.uniform(-3.5, 3.5)
        image = image.rotate(angle, expand=True, fillcolor=bg_color, resample=Image.BICUBIC)

    # 5.2 Gürültü (tarayıcı kirliliği, kağıt dokusu)
    if random.random() > 0.25:
        np_img = np.array(image, dtype=np.int16)
        noise_level = random.randint(3, 12)
        noise = np.random.randint(-noise_level, noise_level, np_img.shape, dtype=np.int16)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(np_img)

    # 5.3 Bulanıklık (odak kaybı, hareket)
    if random.random() > 0.6:
        blur_radius = random.uniform(0.3, 0.9)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 5.4 Kontrast/Parlaklık değişimi
    if random.random() > 0.7:
        np_img = np.array(image, dtype=np.float32)
        # Kontrast
        contrast = random.uniform(0.9, 1.1)
        np_img = np.clip(np_img * contrast, 0, 255)
        # Parlaklık
        brightness = random.randint(-10, 10)
        np_img = np.clip(np_img + brightness, 0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))

    # 6. Kaydet
    image.save(output_path, quality=95)
    return True


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def generate_dataset(font_dir, output_dir, count=10000):
    """
    Kategori bazlı sentetik veri seti üretir.

    Dağılım:
    - %30 Sayılar (3.000)
    - %30 Türkçe Metinler (3.000)
    - %20 Tarihler (2.000)
    - %15 Karışık (1.500)
    - %5  Teknik Kodlar (500)

    Args:
        font_dir: Font klasörü
        output_dir: Çıktı klasörü
        count: Toplam veri sayısı (default: 10.000)

    Returns:
        CSV dosya yolu
    """
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fontları bul
    fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(('.ttf', '.TTF'))]
    if not fonts:
        print("[HATA] Font bulunamadı!")
        return None

    print(f"[Bilgi] {len(fonts)} font bulundu")
    print(f"[Bilgi] {count:,} adet sentetik veri üretiliyor...")

    # Kategori dağılımı
    n_sayilar = int(count * 0.30)
    n_metinler = int(count * 0.30)
    n_tarihler = int(count * 0.20)
    n_karisik = int(count * 0.15)
    n_kodlar = count - n_sayilar - n_metinler - n_tarihler - n_karisik

    print(f"    - Sayılar: {n_sayilar:,}")
    print(f"    - Türkçe Metinler: {n_metinler:,}")
    print(f"    - Tarihler: {n_tarihler:,}")
    print(f"    - Karışık: {n_karisik:,}")
    print(f"    - Teknik Kodlar: {n_kodlar:,}")

    dataset_log = []
    idx = 0

    # 1. Sayılar
    print("\n[1/5] Sayılar üretiliyor...")
    for i in range(n_sayilar):
        text = generate_number()
        font = random.choice(fonts)
        filename = f"syn_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, filename)

        if create_synthetic_image(text, font, output_path):
            dataset_log.append({
                "file_name": filename,
                "text": text,
                "category": "number"
            })
        idx += 1

        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{n_sayilar:,} tamamlandı")

    # 2. Türkçe Metinler
    print("\n[2/5] Türkçe metinler üretiliyor...")
    for i in range(n_metinler):
        text = generate_turkish_text()
        font = random.choice(fonts)
        filename = f"syn_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, filename)

        if create_synthetic_image(text, font, output_path):
            dataset_log.append({
                "file_name": filename,
                "text": text,
                "category": "turkish_text"
            })
        idx += 1

        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{n_metinler:,} tamamlandı")

    # 3. Tarihler
    print("\n[3/5] Tarihler üretiliyor...")
    for i in range(n_tarihler):
        text = generate_date()
        font = random.choice(fonts)
        filename = f"syn_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, filename)

        if create_synthetic_image(text, font, output_path):
            dataset_log.append({
                "file_name": filename,
                "text": text,
                "category": "date"
            })
        idx += 1

        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{n_tarihler:,} tamamlandı")

    # 4. Karışık
    print("\n[4/5] Karışık veriler üretiliyor...")
    for i in range(n_karisik):
        text = generate_mixed()
        font = random.choice(fonts)
        filename = f"syn_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, filename)

        if create_synthetic_image(text, font, output_path):
            dataset_log.append({
                "file_name": filename,
                "text": text,
                "category": "mixed"
            })
        idx += 1

        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{n_karisik:,} tamamlandı")

    # 5. Teknik Kodlar
    print("\n[5/5] Teknik kodlar üretiliyor...")
    for i in range(n_kodlar):
        text = generate_technical_code()
        font = random.choice(fonts)
        filename = f"syn_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, filename)

        if create_synthetic_image(text, font, output_path):
            dataset_log.append({
                "file_name": filename,
                "text": text,
                "category": "code"
            })
        idx += 1

    # CSV oluştur
    df = pd.DataFrame(dataset_log)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"\n{'='*50}")
    print(f"[TAMAMLANDI] {len(df):,} adet veri üretildi!")
    print(f"[Konum] {output_dir}")
    print(f"[CSV] {csv_path}")
    print(f"{'='*50}")

    # İstatistikler
    print("\nKategori Dağılımı:")
    print(df['category'].value_counts())

    return csv_path


# =============================================================================
# DOĞRUDAN ÇALIŞTIRMA
# =============================================================================

if __name__ == "__main__":
    # Test için
    import sys

    # Script'in bulunduğu dizini bul
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    # Varsayılan yollar (mutlak path)
    FONT_DIR = os.path.join(PROJECT_DIR, "data", "fonts")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")

    # Komut satırından count alınabilir
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # Test için 100

    generate_dataset(FONT_DIR, OUTPUT_DIR, count=count)