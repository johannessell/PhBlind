import cv2

def detect_rois(img):
    """Findet kleine Kästchen (ROIs) im Bild."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < w < 60 and 20 < h < 60:
            rois.append((x, y, x + w, y + h))
    rois.sort(key=lambda r: (r[1], r[0]))  # oben→unten, links→rechts
    return rois

def ocr_text_from_roi(img, roi):
    """Dummy OCR – hier kannst du Tesseract oder EasyOCR einbauen."""
    # roi = (x1, y1, x2, y2)
    x1, y1, x2, y2 = roi
    roi_img = img[y1:y2, x1:x2]
    # TODO: OCR einbauen. Momentan nur Dummy.
    return None
