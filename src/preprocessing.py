import cv2
import numpy as np

def correct_skew(image):
    """
    Taranmış belgedeki eğriliği (skew) düzeltir.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[Uyarı] Kontur bulunamadı, düzeltme yapılmıyor.")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > 10:
        print(f"[Bilgi] Açı ({angle:.2f}) çok yüksek, güvenlik gereği düzeltme iptal edildi.")
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    
    return rotated

def find_boxes(image_path, debug=False):
    """
    Form üzerindeki veri kutularını tespit eder.
    Geriye (x, y, w, h) listesi ve işlenmiş resmi döndürür.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    # 1. Eğrilik Düzeltme
    image = correct_skew(image)
    
    # 2. Padding (Çerçeve Ekleme)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    scale = 25
    
    # Yatay ve Dikey Maskeler
    hor_len = np.array(image).shape[1] // scale
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    mask_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    mask_h = cv2.dilate(mask_h, np.ones((1, 5), np.uint8), iterations=1)

    ver_len = np.array(image).shape[0] // scale
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))
    mask_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=1)
    mask_v = cv2.dilate(mask_v, np.ones((5, 1), np.uint8), iterations=1)

    # Tablo İskeleti
    table_mask = cv2.addWeighted(mask_h, 0.5, mask_v, 0.5, 0.0)
    _, table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY)
    
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=2)

    contours, hierarchy = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    box_data = []
    
    if hierarchy is not None:
        for i, c in enumerate(contours):
            if hierarchy[0][i][2] == -1: # En içteki kutu
                x, y, w, h = cv2.boundingRect(c)
                if w > 20 and h > 20:
                    box_data.append({'x':x, 'y':y, 'w':w, 'h':h})

    # Sıralama (Yukarıdan aşağı, soldan sağa)
    box_data = sorted(box_data, key=lambda b: (b['y'], b['x']))

    if debug:
        print(f"Toplam {len(box_data)} kutu bulundu.")
        
    return image, box_data