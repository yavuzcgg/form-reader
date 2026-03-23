import xml.etree.ElementTree as ET
import os
import cv2

def parse_xml_labels(xml_path):
    """LabelImg XML dosyasını okur ve etiketleri döndürür."""
    if not os.path.exists(xml_path):
        print(f"[Hata] XML dosyası bulunamadı: {xml_path}")
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_boxes = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')

        xml_boxes.append({
            'label': name,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        })
    return xml_boxes

def match_and_crop(image, opencv_boxes, xml_boxes, output_folder="data/processed"):
    """
    OpenCV kutuları ile XML etiketlerini eşleştirir ve resimleri keser.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    labeled_data = []
    OFFSET = 10  # Preprocessing'de eklenen padding miktarı

    for idx, box in enumerate(opencv_boxes):
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        # Merkez noktası hesapla
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        
        # Orijinal resme göre merkeze dön
        orig_center_x = center_x - OFFSET
        orig_center_y = center_y - OFFSET

        found_label = None

        for xbox in xml_boxes:
            if (xbox['xmin'] < orig_center_x < xbox['xmax']) and \
               (xbox['ymin'] < orig_center_y < xbox['ymax']):
                found_label = xbox['label']
                break
        
        if found_label:
            # Klasör oluştur
            label_dir = os.path.join(output_folder, found_label)
            os.makedirs(label_dir, exist_ok=True)

            # Resmi kes
            roi_img = image[y:y+h, x:x+w]
            
            # Kaydet
            filename = f"{found_label}_{idx}.jpg"
            save_path = os.path.join(label_dir, filename)
            cv2.imwrite(save_path, roi_img)

            labeled_data.append({
                'label': found_label,
                'file_path': save_path,
                'roi_image': roi_img
            })

    return labeled_data