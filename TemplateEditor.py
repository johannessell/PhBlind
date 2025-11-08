import cv2
import json
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock


# ---------------- ROI-Erkennung ----------------
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
    x1, y1, x2, y2 = roi
    roi_img = img[y1:y2, x1:x2]
    # TODO: OCR einbauen. Momentan nur Dummy.
    return None


# ---------------- Template Editor ----------------
class TemplateEditor(BoxLayout):
    def __init__(self, template_img, on_done, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.template_img = template_img
        self.on_done = on_done

        self.rois = detect_rois(template_img)
        self.ph_values = []

        # OCR-Versuch für jeden ROI
        for roi in self.rois:
            candidates = []
            for dx in [5, -50]:
                x1, y1, x2, y2 = roi
                text_rect = (
                    max(0, x2 + dx),
                    y1,
                    min(template_img.shape[1], x2 + dx + 50),
                    y2,
                )
                val = ocr_text_from_roi(template_img, text_rect)
                if val is not None:
                    candidates.append(val)
            self.ph_values.append(candidates[0] if candidates else 7.0)

        # Bild-Anzeige
        self.img_widget = Image(size_hint=(0.6, 1))
        self.add_widget(self.img_widget)
        self.update_view()

        # Eingabefelder rechts
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
        # ROIs + Werte einzeichnen
        for i, (x1, y1, x2, y2) in enumerate(self.rois):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, str(self.ph_values[i]),
                (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(annotated, 0).tobytes()
        texture = Texture.create(size=(annotated.shape[1], annotated.shape[0]), colorfmt="rgb")
        texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        self.img_widget.texture = texture

    def finish(self, *args):
        self.ph_values = []
        for inp in self.inputs:
            try:
                self.ph_values.append(float(inp.text))
            except ValueError:
                self.ph_values.append(7.0)

        config = {"rois": self.rois, "ph_values": self.ph_values}

        # Dateiname mit Zeitstempel
        fname = f"template_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Template gespeichert: {fname}")
        self.on_done()


# ---------------- Demo-App mit Live-Kamera ----------------
class LiveView(BoxLayout):
    """Zeigt den Live-Kamerastream + Button zum Aufnehmen."""
    def __init__(self, app, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.app = app

        # Kamera öffnen
        self.capture = cv2.VideoCapture(0)

        # Anzeige
        self.img_widget = Image(size_hint=(1, 0.9))
        self.add_widget(self.img_widget)

        # Button
        btn = Button(text="📷 Snapshot aufnehmen", size_hint=(1, 0.1))
        btn.bind(on_press=self.take_snapshot)
        self.add_widget(btn)

        # Update-Loop für Kamera
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(frame_rgb, 0).tobytes()
            texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt="rgb")
            texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
            self.img_widget.texture = texture
            self.last_frame = frame.copy()

    def take_snapshot(self, *args):
        if hasattr(self, "last_frame"):
            self.capture.release()
            self.app.open_template_editor(self.last_frame)


class DemoApp(App):
    def build(self):
        self.root = BoxLayout(orientation="vertical")
        self.live = LiveView(self)
        self.root.add_widget(self.live)
        return self.root

    def open_template_editor(self, frame):
        self.root.clear_widgets()
        self.editor = TemplateEditor(frame, self.back_to_live)
        self.root.add_widget(self.editor)

    def back_to_live(self):
        self.root.clear_widgets()
        self.live = LiveView(self)
        self.root.add_widget(self.live)


if __name__ == "__main__":
    DemoApp().run()
