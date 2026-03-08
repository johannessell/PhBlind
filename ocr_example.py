import os
import cv2
import easyocr
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.core.window import Window

class OCRApp(App):
    def build(self):
        # EasyOCR Reader initialisieren
        self.reader = easyocr.Reader(['de', 'en'], gpu=True)

        # Bilder im Ordner "template" sammeln
        template_dir = "templates"
        self.image_files = [os.path.join(template_dir, f) for f in os.listdir(template_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        self.image_files.sort()

        self.current_index = 0

        # Image-Widget für Anzeige
        self.image_widget = Image()
        self.update_image()

        # Tastenevents für Blättern
        Window.bind(on_key_down=self.on_key_down)

        # Layout
        layout = BoxLayout()
        layout.add_widget(self.image_widget)
        return layout

    def update_image(self):
        """Aktuelles Bild OCR-verarbeiten und im Kivy-Image-Widget anzeigen"""
        if not self.image_files:
            return

        path = self.image_files[self.current_index]
        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert colors if text is dark on light background
        gray = cv2.bitwise_not(gray)


        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))

        # Get rotated rectangle of the text
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        print(f"Detected angle: {angle} degrees")

        # OCR-Erkennung
        results = self.reader.readtext(img, rotation_info=[0] , allowlist="0123456789,phPHOombMBgGlL/.")

        # Bounding Boxes + Text + Confidence einzeichnen
        for (bbox, text, prob) in results:
            pts = [(int(x), int(y)) for x, y in bbox]
            cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

            label = f"{text} ({prob:.2f})"
            x, y = pts[0]
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 2, cv2.LINE_AA)

        # Bild in Texture umwandeln
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buf = img_rgb.tobytes()
        texture = Texture.create(size=(img_rgb.shape[1], img_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()

        # Texture im Kivy-Image setzen
        self.image_widget.texture = texture

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        """Mit Pfeiltasten durch Bilder blättern"""
        if key == 275:  # Pfeil rechts
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.update_image()
        elif key == 276:  # Pfeil links
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.update_image()

if __name__ == "__main__":
    OCRApp().run()
