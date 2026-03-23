"""
OCR Engine - TÜBİTAK 3501 Projesi
==================================
TrOCR tabanlı Türkçe el yazısı tanıma motoru.

Desteklenen model formatları:
1. Deney #003: TrOCR Encoder + BERTurk Decoder (TrOCR processor + BERTurk tokenizer)
2. Deney #002: ViT + BERTurk (ViT processor + BERTurk tokenizer)
3. Base: microsoft/trocr-base-handwritten (TrOCRProcessor)
"""

from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    TrOCRProcessor
)
from PIL import Image
import torch
import os
import json
import glob


class OCREngine:
    """
    TrOCR tabanlı OCR motoru.

    Kullanım:
        ocr = OCREngine()
        text = ocr.predict("image.jpg")
    """

    def __init__(self, model_path=None):
        # Varsayılan model yolları
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        finetuned_path = os.path.join(project_dir, "models", "trocr-turkish-handwritten")
        base_model = "microsoft/trocr-base-handwritten"

        # Model seçimi
        if model_path:
            selected_model = model_path
        else:
            # Deney klasörlerini ara (deney001, deney002, ...)
            deney_dirs = sorted(glob.glob(os.path.join(finetuned_path, "deney*")))
            if deney_dirs:
                selected_model = deney_dirs[-1]  # En son deney
                print(f"[Bilgi] Fine-tuned model bulundu: {os.path.basename(selected_model)}")
            elif os.path.exists(os.path.join(finetuned_path, "config.json")):
                selected_model = finetuned_path  # Eski format (versiyonlama öncesi)
                print(f"[Bilgi] Fine-tuned Türkçe model bulundu!")
            else:
                selected_model = base_model
                print(f"[Uyarı] Fine-tuned model bulunamadı, base model kullanılıyor.")

        print(f"[Bilgi] OCR Modeli Yükleniyor... ({selected_model})")

        # Cihaz seçimi
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Bilgi] Cihaz: {self.device.upper()}")

        # Model tipini belirle ve yükle
        self._load_model(selected_model, base_model)

        # Evaluation moduna al
        self.model.eval()
        print("[Bilgi] Model hazır!")

    def _load_model(self, model_path, base_model):
        """
        Modeli ve processor'ları yükler.
        Format algılama config.json'daki decoder.model_type'a göre yapılır.
        """
        is_local = os.path.exists(model_path)

        if is_local:
            config_path = os.path.join(model_path, "config.json")
            tokenizer_config = os.path.join(model_path, "tokenizer_config.json")
            has_tokenizer = os.path.exists(tokenizer_config)

            # config.json'dan decoder tipini oku
            decoder_type = None
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                decoder_type = config.get("decoder", {}).get("model_type", None)

            if decoder_type == "bert" and has_tokenizer:
                # DENEY #003: TrOCR Encoder + BERTurk Decoder
                print("[Bilgi] TrOCR Encoder + BERTurk Decoder formatı algılandı (Deney #003)")

                self.processor = ViTImageProcessor.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
                self.use_separate_processors = True
                self.format = "TrOCR Encoder + BERTurk Decoder"

            elif has_tokenizer and decoder_type in ("trocr", None):
                # DENEY #002 veya eski format: ViT/TrOCR encoder + BERTurk tokenizer
                print("[Bilgi] ViT + BERTurk formatı algılandı (Deney #002)")

                self.processor = ViTImageProcessor.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
                self.use_separate_processors = True
                self.format = "ViT + BERTurk"

            else:
                # Standart TrOCR formatı
                print("[Bilgi] TrOCR formatı algılandı")
                self._load_trocr_format(model_path, base_model)
                return
        else:
            # HuggingFace'den yükle (base model)
            self._load_trocr_format(model_path, base_model)
            return

        # Tokenizer test
        test_tokens = self.tokenizer.tokenize("Üretim Şişe")
        print(f"[Bilgi] Tokenizer test: {test_tokens}")

    def _load_trocr_format(self, model_path, base_model):
        """Standart TrOCRProcessor formatı ile yükler."""
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path)
        except:
            self.processor = TrOCRProcessor.from_pretrained(base_model)

        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        except:
            self.model = VisionEncoderDecoderModel.from_pretrained(base_model).to(self.device)

        self.use_separate_processors = False
        self.tokenizer = None
        self.format = "TrOCR (base)"

    def predict(self, image_input, return_confidence=False):
        """
        Görüntüden metin okur.

        Args:
            image_input: Dosya yolu (str), PIL Image veya numpy array
            return_confidence: True ise confidence score da döndür

        Returns:
            str veya (str, float)
        """
        try:
            # Görüntü yükleme
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                import cv2
                rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_image)

            # Görüntüyü işle
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            # Tahmin yap
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=2.0,
                    repetition_penalty=1.5,
                    return_dict_in_generate=return_confidence,
                    output_scores=return_confidence,
                )

            # Decode
            if return_confidence:
                sequences = generated_ids.sequences
                if hasattr(generated_ids, 'sequences_scores') and generated_ids.sequences_scores is not None:
                    confidence = torch.exp(generated_ids.sequences_scores[0]).item()
                else:
                    confidence = 0.0

                if self.use_separate_processors:
                    text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
                else:
                    text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]
                return text, confidence
            else:
                if self.use_separate_processors:
                    text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return text

        except Exception as e:
            print(f"[Hata] OCR okuma hatası: {e}")
            if return_confidence:
                return "", 0.0
            return ""

    def predict_batch(self, image_inputs):
        """Birden fazla görüntüden metin okur."""
        results = []
        for img in image_inputs:
            results.append(self.predict(img))
        return results

    def get_model_info(self):
        """Model bilgilerini döndürür."""
        info = {
            "device": self.device,
            "format": self.format,
            "vocab_size": self.model.decoder.config.vocab_size,
            "max_length": 64,
            "num_beams": 4,
        }

        if self.use_separate_processors and self.tokenizer:
            info["tokenizer_test"] = self.tokenizer.tokenize("Ü ş ı ğ ö ç")

        return info


# Backward compatibility alias
TrOCREngine = OCREngine


# =============================================================================
# DOĞRUDAN ÇALIŞTIRMA (Test)
# =============================================================================

if __name__ == "__main__":
    import sys

    ocr = OCREngine()

    print("\nModel Bilgisi:")
    for k, v in ocr.get_model_info().items():
        print(f"  {k}: {v}")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n[Test] Görüntü: {image_path}")

        text, confidence = ocr.predict(image_path, return_confidence=True)
        print(f"[Sonuç] Metin: {text}")
        print(f"[Sonuç] Confidence: {confidence:.2%}")
    else:
        print("\nKullanım: python ocr_engine.py <görüntü_yolu>")
