import cv2
import numpy as np
import json
import easyocr
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from sklearn.linear_model import LinearRegression

# ---------------- Einstellungen ----------------
DEBUG = True
MIN_MATCH_COUNT = 50
MIN_INLIER_COUNT = 25
MIN_AREA = 500

# ---------------- OCR Setup ----------------
reader = easyocr.Reader(['en'])

# ---------------- ΔE & Lab-Funktionen ----------------
def median_color_in_rect(img_bgr, rect):
    x1, y1, x2, y2 = rect
    x1, x2 = int(round(min(x1, x2))), int(round(max(x1, x2)))
    y1, y2 = int(round(min(y1, y2))), int(round(max(y1, y2)))
    h, w = img_bgr.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0)
    patch_bgr = img_bgr[y1:y2, x1:x2]
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2Lab)
    med = np.median(patch_lab.reshape(-1, 3), axis=0)
    return float(med[0]), float(med[1]), float(med[2])

def opencv_lab_to_cie_lab(lab):
    L, a, b = lab
    return L * 100 / 255, a - 128, b - 128

def delta_e_ciede2000(lab1, lab2):
    L1,a1,b1 = lab1
    L2,a2,b2 = lab2
    avg_L = (L1+L2)/2
    C1 = np.sqrt(a1**2+b1**2)
    C2 = np.sqrt(a2**2+b2**2)
    avg_C = (C1+C2)/2
    G = 0.5*(1-np.sqrt((avg_C**7)/(avg_C**7+25**7)))
    a1p=(1+G)*a1
    a2p=(1+G)*a2
    C1p=np.sqrt(a1p**2+b1**2)
    C2p=np.sqrt(a2p**2+b2**2)
    dhp = np.degrees(np.arctan2(b2,a2p)) - np.degrees(np.arctan2(b1,a1p))
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2)
    dLp = L2-L1
    dCp = C2p-C1p
    return np.sqrt(dLp**2 + dCp**2 + dHp**2)  # einfache Variante

# ---------------- OCR-Funktion ----------------
def ocr_text_from_roi(img, rect):
    x1, y1, x2, y2 = rect
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    results = reader.readtext(roi, detail=0)
    for text in results:
        try:
            return float(text.strip().replace(",", "."))
        except:
            continue
    return None



# ---------------- Live PH-App ----------------
class LivePHApp(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = "vertical"
        self.img_widget = Image()
        self.add_widget(self.img_widget)
        self.label = Label(text='Suche Indikator...', size_hint_y=None, height=40)
        self.add_widget(self.label)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1/30)

        # Lade Template
        try:
            with open("template.json", "r") as f:
                template = json.load(f)
            self.rois = template["rois"]
            self.ref_pH = np.array(template["ph_values"])
        except:
            self.rois = []
            self.ref_pH = np.array([7.0]*10)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        annotated = frame.copy()
        medians = []
        for r in self.rois:
            med = median_color_in_rect(frame, r)
            medians.append(opencv_lab_to_cie_lab(med))

        if len(medians) > 0:
            # Referenz aus median der ersten Hälfte
            y_L = np.array([lab[0] for lab in medians])
            y_a = np.array([lab[1] for lab in medians])
            y_b = np.array([lab[2] for lab in medians])

            # Linear Fit
            mask = (self.ref_pH >= 6.8) & (self.ref_pH <= 8.2)
            x_lin = self.ref_pH[mask].reshape(-1, 1)
            y_L_lin, y_a_lin, y_b_lin = y_L[mask], y_a[mask], y_b[mask]
            model_L = LinearRegression().fit(x_lin, y_L_lin)
            model_a = LinearRegression().fit(x_lin, y_a_lin)
            model_b = LinearRegression().fit(x_lin, y_b_lin)
            x_fine = np.linspace(self.ref_pH.min(), self.ref_pH.max(), 1000)
            y_L_fit = model_L.predict(x_fine.reshape(-1, 1))
            y_a_fit = model_a.predict(x_fine.reshape(-1, 1))
            y_b_fit = model_b.predict(x_fine.reshape(-1, 1))

            # ΔE
            errors = [delta_e_ciede2000((L, a, b), (y_L[0], y_a[0], y_b[0])) for L, a, b in zip(y_L_fit, y_a_fit, y_b_fit)]
            best_idx = np.argmin(errors)
            estimated_pH = x_fine[best_idx]
            self.label.text = f"Geschätzter pH: {estimated_pH:.2f}"

        # Annotiertes Bild
        for i, (r, lab) in enumerate(zip(self.rois, medians)):
            x1, y1, x2, y2 = r
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def capture_template(self):
        ret, frame = self.capture.read()
        if ret:
            self.app.open_template_editor(frame)

# ---------------- Haupt-App ----------------
class MyApp(App):
    def build(self):
        self.root_widget = BoxLayout(orientation="vertical")
        self.live = LivePHApp(self)
        self.root_widget.add_widget(self.live)
        btn_template = Button(text="Template aufnehmen", size_hint_y=None, height=40)
        btn_template.bind(on_press=lambda _: self.live.capture_template())
        self.root_widget.add_widget(btn_template)
        return self.root_widget

    def open_template_editor(self, frame):
        self.root_widget.clear_widgets()
        self.editor = TemplateEditor(frame, self.back_to_live)
        self.root_widget.add_widget(self.editor)

    def back_to_live(self):
        self.root_widget.clear_widgets()
        self.live = LivePHApp(self)
        self.root_widget.add_widget(self.live)
        btn_template = Button(text="Template aufnehmen", size_hint_y=None, height=40)
        btn_template.bind(on_press=lambda _: self.live.capture_template())
        self.root_widget.add_widget(btn_template)

if __name__ == "__main__":
    MyApp().run()
