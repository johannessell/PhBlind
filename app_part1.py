import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# ---------------- Einstellungen ----------------
DEBUG = True   # Wenn True → Keypoints anzeigen & Debug-Ausgaben
MIN_MATCH_COUNT = 50
MIN_INLIER_COUNT = 25
MIN_AREA = 500

# ---------------- Template vorbereiten ----------------
template_color = cv2.imread('template03.jpg')
if template_color is None:
    raise FileNotFoundError("Template konnte nicht geladen werden! Prüfe den Pfad.")

template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

# SIFT Initialisierung
sift = cv2.SIFT_create()
# sift = cv2.ORB_create(2000)
# sift = cv2.AKAZE_create()
kp_template, des_template = sift.detectAndCompute(template_gray, None)

# Debuganzeige
if DEBUG:
    template_drawn = cv2.drawKeypoints(template_gray, kp_template, None, color=(0, 255, 0))
    cv2.imshow('Template Features', template_drawn)

# Matcher vorbereiten
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# ---------------- Kivy App ----------------
class PHApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.img_widget = Image()
        self.add_widget(self.img_widget)
        self.label = Label(text='Suche Indikator...', size_hint_y=None, height=40)
        self.add_widget(self.label)

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden")

        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Frame in Grauwert umwandeln
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)

        # Für Debug Keypoints zeichnen
        if DEBUG:
            draw_frame = cv2.drawKeypoints(frame_gray, kp_frame, None, color=(0, 255, 0))
        else:
            draw_frame = frame.copy()

        if des_frame is not None and len(des_frame) > 0:
            matches = bf.knnMatch(des_template, des_frame, k=2)

            # Lowe’s Ratio-Test
            good = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if DEBUG:
                print(f"number of good matches: {len(good)}")

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    inliers = int(np.sum(mask))
                    if inliers > MIN_INLIER_COUNT:
                        h, w = template_gray.shape[:2]
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        area = cv2.contourArea(np.int32(dst))
                        if area > MIN_AREA:
                            draw_frame = cv2.polylines(draw_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                            self.label.text = f'Indikator erkannt! ({inliers} Inlier)'
                        else:
                            self.label.text = 'Zu kleiner Bereich → ignoriert'
                    else:
                        self.label.text = 'Zu wenige Inlier → kein Treffer'
                else:
                    self.label.text = 'Homographie fehlgeschlagen'
            else:
                self.label.text = 'Zu wenige gute Matches'
        else:
            self.label.text = 'Keine Features gefunden'

        # OpenCV BGR → RGB für Kivy
        frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)

        # Texture für Kivy erzeugen
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture


class MyApp(App):
    def build(self):
        return PHApp()


if __name__ == '__main__':
    MyApp().run()
