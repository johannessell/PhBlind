import cv2
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# ---------------- Einstellungen ----------------
DEBUG = True
TEMPLATE_FILE = "template.jpg"
ROI_CONFIG_FILE = "template.json"

MIN_MATCH_COUNT = 50
MIN_INLIER_COUNT = 25
MIN_AREA = 500

# ---------------- Hilfsfunktionen ----------------
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
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if dhp > 180:
        dhp -= 360
    elif dhp < -180:
        dhp += 360
    elif C1p * C2p == 0:
        dhp = 0
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    avg_Lp = (L1 + L2) / 2
    avg_Cp = (C1p + C2p) / 2
    avg_hp = (h1p + h2p + 360) / 2 if abs(h1p - h2p) > 180 else (h1p + h2p) / 2
    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.2 * np.cos(np.radians(4 * avg_hp - 63))
    )
    d_ro = 30 * np.exp(-((avg_hp - 275) / 25) ** 2)
    RC = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    SL = 1 + 0.015 * (avg_Lp - 50) ** 2 / np.sqrt(20 + (avg_Lp - 50) ** 2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T
    RT = -np.sin(2 * np.radians(d_ro)) * RC
    return np.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)
    )

def fit_ph_curve(ref_pH, sample_rois, ref_roi):
    """Lineares Fitting + ΔE-Matching → pH-Schätzung"""
    y_L = np.array([lab[0] for lab in sample_rois])
    y_a = np.array([lab[1] for lab in sample_rois])
    y_b = np.array([lab[2] for lab in sample_rois])
    y_L_ref, y_a_ref, y_b_ref = ref_roi

    mask = (ref_pH >= 6.8) & (ref_pH <= 8.2)
    x_lin = ref_pH[mask].reshape(-1, 1)
    y_L_lin, y_a_lin, y_b_lin = y_L[mask], y_a[mask], y_b[mask]

    model_L = LinearRegression().fit(x_lin, y_L_lin)
    model_a = LinearRegression().fit(x_lin, y_a_lin)
    model_b = LinearRegression().fit(x_lin, y_b_lin)

    x_fine = np.linspace(ref_pH.min(), ref_pH.max(), 1000)
    y_L_fit = model_L.predict(x_fine.reshape(-1, 1))
    y_a_fit = model_a.predict(x_fine.reshape(-1, 1))
    y_b_fit = model_b.predict(x_fine.reshape(-1, 1))

    errors = [
        delta_e_ciede2000((L, a, b), (y_L_ref, y_a_ref, y_b_ref))
        for L, a, b in zip(y_L_fit, y_a_fit, y_b_fit)
    ]
    best_idx = np.argmin(errors)
    return x_fine[best_idx]

def detect_rois(template_img, min_area=200, max_area=5000):
    """Automatische ROI-Erkennung via Konturen"""
    gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min_area < w * h < max_area:
            rois.append((x, y, x + w, y + h))

    # sortieren: erst nach y (Zeilen), dann nach x (Spalten)
    rois = sorted(rois, key=lambda r: (r[1], r[0]))
    return rois

# ---------------- Template Editor ----------------
class TemplateEditor(BoxLayout):
    def __init__(self, template_img, on_done, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.template_img = template_img
        self.on_done = on_done

        # Automatische ROI-Erkennung
        self.rois = detect_rois(template_img)
        self.ph_values = [7.0] * len(self.rois)  # Defaultwerte

        # Linke Seite: Bild mit ROIs
        self.img_widget = Image(size_hint=(0.6, 1))
        self.add_widget(self.img_widget)
        self.update_view()

        # Rechte Seite: Eingabefelder für Werte
        right_panel = BoxLayout(orientation="vertical", size_hint=(0.4, 1))
        self.inputs = []
        grid = GridLayout(cols=2, size_hint=(1, 0.9))
        for i, roi in enumerate(self.rois):
            grid.add_widget(Label(text=f"ROI {i+1}"))
            ti = TextInput(text=str(self.ph_values[i]), multiline=False)
            self.inputs.append(ti)
            grid.add_widget(ti)
        right_panel.add_widget(grid)

        btn_save = Button(text="Speichern & zurück", size_hint=(1, 0.1))
        btn_save.bind(on_press=self.finish)
        right_panel.add_widget(btn_save)

        self.add_widget(right_panel)

    def update_view(self):
        annotated = self.template_img.copy()
        for (x1, y1, x2, y2) in self.rois:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        buf = cv2.flip(annotated, 0).tobytes()
        texture = Texture.create(size=(annotated.shape[1], annotated.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def finish(self, *args):
        self.ph_values = [float(inp.text) for inp in self.inputs]
        config = {"rois": self.rois, "ph_values": self.ph_values}
        with open(ROI_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        self.on_done()

# ---------------- Live-Modus ----------------
class LivePHApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.img_widget = Image()
        self.add_widget(self.img_widget)
        self.label = Label(text="Starte Kamera...", size_hint_y=None, height=40)
        self.add_widget(self.label)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30)

        self.ref_pH = np.array([8.2, 7.8, 7.6, 7.4, 7.2, 7.0, 6.8])

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # hier nur Live-Anzeige (vereinfachte Demo)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def capture_template(self):
        """Template-Bild aufnehmen und Editor öffnen"""
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite(TEMPLATE_FILE, frame)
            self.parent.open_template_editor(frame)

# ---------------- Main App ----------------
class MyApp(App):
    def build(self):
        self.root_widget = BoxLayout(orientation="vertical")
        self.live = LivePHApp()
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
        self.live = LivePHApp()
        self.root_widget.add_widget(self.live)
        btn_template = Button(text="Template aufnehmen", size_hint_y=None, height=40)
        btn_template.bind(on_press=lambda _: self.live.capture_template())
        self.root_widget.add_widget(btn_template)

if __name__ == "__main__":
    MyApp().run()
