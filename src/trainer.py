"""
TrOCR Fine-Tuning - TÜBİTAK 3501 Projesi
=========================================
Deney #005: TrOCR + Türkçe Tokenizer (Smart Init + NFC + Encoder Freeze)

3 kanıtlanmış iyileştirme:
1. Smart Embedding Init: 'ş' için 's' ağırlığını kopyala (VIPI Paper, arxiv 2112.14569)
2. Unicode NFC Normalization: Türkçe karakter tutarlılığı (HuggingFace Issue #6680)
3. Encoder Freeze: ViT encoder dondur, sadece decoder Türkçe öğrensin (DLoRA-TrOCR, 2024)
"""

import os
import unicodedata
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SABİTLER
# =============================================================================

BASE_MODEL = "microsoft/trocr-base-handwritten"

TURKISH_CHARS = ['ş', 'ğ', 'ü', 'ı', 'ö', 'ç', 'Ş', 'Ğ', 'Ü', 'İ', 'Ö', 'Ç']

# Smart Init: Türkçe → İngilizce benzer karakter eşleştirmesi
CHAR_MAPPING = {
    'ş': 's', 'ğ': 'g', 'ü': 'u', 'ı': 'i', 'ö': 'o', 'ç': 'c',
    'Ş': 'S', 'Ğ': 'G', 'Ü': 'U', 'İ': 'I', 'Ö': 'O', 'Ç': 'C'
}


# =============================================================================
# SMART EMBEDDING INITIALIZATION
# =============================================================================

def smart_initialize_embeddings(model, tokenizer):
    """
    Yeni eklenen Türkçe karakterlerin embedding ağırlıklarını,
    İngilizce benzerlerinden kopyalar (warm start).

    Referans: VIPI Paper (arxiv 2112.14569)
    'ş' → 's' ağırlığı ile başlar, sıfırdan değil benzeren öğrenir.
    """
    embeddings = model.decoder.get_input_embeddings()

    for turk_char, eng_char in CHAR_MAPPING.items():
        turk_id = tokenizer.convert_tokens_to_ids(turk_char)
        eng_id = tokenizer.convert_tokens_to_ids(eng_char)

        if turk_id != tokenizer.unk_token_id and eng_id != tokenizer.unk_token_id:
            with torch.no_grad():
                embeddings.weight[turk_id] = embeddings.weight[eng_id].clone()
            print(f"      [Init] '{turk_char}' ← '{eng_char}' ağırlıkları kopyalandı")

    # Output embedding (lm_head) da güncelle
    if hasattr(model.decoder, 'output') and hasattr(model.decoder.output, 'dense'):
        pass  # TrOCR'da output embedding tied olabilir, ayrı güncelleme gerekmez

    print("      [OK] Smart Embedding Initialization tamamlandı")


# =============================================================================
# DATASET
# =============================================================================

class TurkishHandwritingDataset(Dataset):
    """
    Türkçe el yazısı veri seti.
    Unicode NFC normalization uygulanır (HuggingFace Issue #6680 düzeltmesi).
    """
    def __init__(self, root_dir, df, processor, max_length=64):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_name']
        text = str(self.df.iloc[idx]['text'])

        # Unicode NFC Normalization - Türkçe karakter tutarlılığı
        # "ş" NFC'de tek karakter, NFD'de "s" + birleştirme işareti olur
        text = unicodedata.normalize('NFC', text)

        image_path = os.path.join(self.root_dir, file_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Uyarı] Görüntü yüklenemedi: {image_path} - {e}")
            image = Image.new("RGB", (384, 384), color=(255, 255, 255))

        # Görüntü işleme (TrOCR processor - 384x384)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Etiket tokenize (RoBERTa tokenizer + Türkçe genişletme)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        ).input_ids

        # Padding tokenlarını -100 yap (loss hesabından düşür)
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }


# =============================================================================
# ANA EĞİTİM FONKSİYONU
# =============================================================================

def train_trocr(
    dataset_dir,
    csv_path,
    output_model_dir="models/trocr-turkish-handwritten",
    epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    resume_from_checkpoint=None
):
    """
    Deney #005: TrOCR + Türkçe Tokenizer (Smart Init + NFC + Encoder Freeze)

    3 iyileştirme:
    1. Smart Embedding Init: Türkçe karakter ağırlıkları İngilizce benzerlerinden kopyalanır
    2. Unicode NFC: Veri tutarlılığı sağlanır
    3. Encoder Freeze: ViT encoder dondurulur, sadece decoder Türkçe öğrenir
    """

    print("=" * 60)
    print("Deney #005: TrOCR + Türkçe Tokenizer")
    print("(Smart Init + NFC + Encoder Freeze)")
    print(f"Model: {BASE_MODEL}")
    print("=" * 60)

    # GPU Kontrolü
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("[UYARI] GPU bulunamadı! CPU ile eğitim çok yavaş olacak.")

    # 1. Veri Setini Hazırla
    print(f"\n[1/5] Veri seti yükleniyor...")
    df = pd.read_csv(csv_path)
    print(f"      Toplam veri: {len(df):,}")

    # %90 Eğitim, %10 Validation
    train_df = df.sample(frac=0.9, random_state=42)
    eval_df = df.drop(train_df.index)
    print(f"      Eğitim: {len(train_df):,} | Validation: {len(eval_df):,}")

    # 2. Processor ve Model Yükle
    print(f"\n[2/5] Model ve Tokenizer yükleniyor...")

    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # Türkçe karakterleri tokenizer'a TEK TOKEN olarak ekle
    num_added = processor.tokenizer.add_tokens(TURKISH_CHARS)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"      Tokenizer'a {num_added} Türkçe karakter eklendi")
    print(f"      Yeni vocab size: {len(processor.tokenizer):,}")

    # Tokenizer test
    test_text = "başarılı Üretim Şişe ğüzel"
    test_tokens = processor.tokenizer.tokenize(test_text)
    print(f"      Tokenizer test: '{test_text}'")
    print(f"      Tokens: {test_tokens}")

    # 3. Smart Embedding Initialization
    print(f"\n[3/5] Smart Embedding Initialization...")
    smart_initialize_embeddings(model, processor.tokenizer)

    # Token ID'lerini ayarla
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # 4. Encoder Freeze
    print(f"\n[4/5] Encoder dondurulıyor...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Decoder'da gradient checkpointing
    if hasattr(model.decoder, 'gradient_checkpointing_enable'):
        model.decoder.gradient_checkpointing_enable()

    # GPU'ya taşı
    model = model.to(device)

    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"      Toplam parametre: {total_params/1e6:.1f}M")
    print(f"      Eğitilebilir (decoder): {trainable_params/1e6:.1f}M")
    print(f"      Dondurulmuş (encoder): {frozen_params/1e6:.1f}M")
    print(f"      [OK] Encoder FROZEN, sadece decoder eğitilecek")

    # 5. Dataset ve Eğitim
    print(f"\n[5/5] Eğitim ayarları yapılandırılıyor...")

    train_ds = TurkishHandwritingDataset(dataset_dir, train_df, processor)
    eval_ds = TurkishHandwritingDataset(dataset_dir, eval_df, processor)

    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch  # 1 epoch warmup
    eval_steps = steps_per_epoch // 2
    save_steps = steps_per_epoch

    print(f"      Epochs: {epochs}")
    print(f"      Batch size: {batch_size}")
    print(f"      Learning rate: {learning_rate}")
    print(f"      Steps/epoch: {steps_per_epoch:,}")
    print(f"      Toplam steps: {total_steps:,}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,

        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,

        fp16=torch.cuda.is_available(),

        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=8,
        predict_with_generate=False,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=False,

        logging_dir=os.path.join(output_model_dir, "logs"),
        logging_steps=50,
        report_to="none",

        dataloader_num_workers=0,
        dataloader_pin_memory=True if device == "cuda" else False,
        gradient_accumulation_steps=2,

        remove_unused_columns=False,
        label_names=["labels"],
    )

    print("\n" + "-" * 60)
    print("Eğitim başlıyor...")
    print("-" * 60)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n[Bilgi] Eğitim kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Eğitim hatası: {e}")
        raise

    # Modeli Kaydet
    print("\n" + "=" * 60)
    print("[Kayıt] Model kaydediliyor...")

    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)

    print(f"[OK] Model kaydedildi: {output_model_dir}")
    print("=" * 60)

    return output_model_dir


# =============================================================================
# DOĞRUDAN ÇALIŞTIRMA
# =============================================================================

if __name__ == "__main__":
    import sys

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    DATASET_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")
    CSV_PATH = os.path.join(PROJECT_DIR, "data", "synthetic", "metadata.csv")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "trocr-turkish-handwritten")

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    experiment = sys.argv[3] if len(sys.argv) > 3 else "deney005"

    train_trocr(
        dataset_dir=DATASET_DIR,
        csv_path=CSV_PATH,
        output_model_dir=os.path.join(OUTPUT_DIR, experiment),
        epochs=epochs,
        batch_size=batch_size
    )
