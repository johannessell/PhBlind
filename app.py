import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ---------------- Einstellungen ----------------
DEBUG = True   # Wenn True → Keypoints anzeigen & Debug-Ausgaben
MIN_MATCH_COUNT = 50
MIN_INLIER_COUNT = 25
MIN_AREA = 500


# ---------------- ΔE & Lab-Funktionen ----------------
def median_color_in_rect(img_bgr, rect):
    x1,y1,x2,y2 = rect
    x1, x2 = int(round(min(x1,x2))), int(round(max(x1,x2)))
    y1, y2 = int(round(min(y1,y2))), int(round(max(y1,y2)))
    h, w = img_bgr.shape[:2]
    x1, x2 = max(0,x1), min(w,x2)
    y1, y2 = max(0,y1), min(h,y2)
    if x2<=x1 or y2<=y1: return (0,0,0)
    patch_bgr = img_bgr[y1:y2, x1:x2]
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2Lab)
    med = np.median(patch_lab.reshape(-1,3), axis=0)
    return float(med[0]), float(med[1]), float(med[2])

def opencv_lab_to_cie_lab(lab):
    L,a,b = lab
    return L*100/255, a-128, b-128

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
    avg_Cp=(C1p+C2p)/2
    h1p = np.degrees(np.arctan2(b1,a1p))%360
    h2p = np.degrees(np.arctan2(b2,a2p))%360
    dLp=L2-L1
    dCp=C2p-C1p
    dhp=h2p-h1p
    if dhp>180: dhp-=360
    elif dhp<-180: dhp+=360
    elif C1p*C2p==0: dhp=0
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2)
    avg_Lp=(L1+L2)/2
    avg_Cp=(C1p+C2p)/2
    avg_hp = (h1p+h2p+360)/2 if abs(h1p-h2p)>180 else (h1p+h2p)/2
    T=(1-0.17*np.cos(np.radians(avg_hp-30))
       +0.24*np.cos(np.radians(2*avg_hp))
       +0.32*np.cos(np.radians(3*avg_hp+6))
       -0.2*np.cos(np.radians(4*avg_hp-63)))
    d_ro = 30*np.exp(-((avg_hp-275)/25)**2)
    RC = 2*np.sqrt((avg_Cp**7)/(avg_Cp**7+25**7))
    SL = 1+0.015*(avg_Lp-50)**2/np.sqrt(20+(avg_Lp-50)**2)
    SC = 1+0.045*avg_Cp
    SH = 1+0.015*avg_Cp*T
    RT = -np.sin(2*np.radians(d_ro))*RC
    return np.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))

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
        self.orientation='vertical'
        self.img_widget=Image()
        self.add_widget(self.img_widget)
        self.label=Label(text='Suche Indikator...',size_hint_y=None,height=40)
        self.add_widget(self.label)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30)

        self.ref_pH = np.array([8.2,7.8,7.6,7.4,7.2,7.0,6.8])

    def update(self,dt):



        ret, frame = self.capture.read()
        if not ret: return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
        kp_frame, des_frame = sift.detectAndCompute(gray,None)
        if des_frame is not None:
            matches = bf.knnMatch(des_template, des_frame, k=2)
            good = [m for m,n in matches if m.distance<0.75*n.distance]
            if len(good)>4:
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
                h,w = template.shape
                warped = cv2.warpPerspective(frame,np.linalg.inv(M),(w,h))

                # ---------------- ROI-Aufteilung ----------------
                cols, rows = 2, 8
                offset_x, offset_y = 95, 50+100+45
                pad_x, pad_y = 50, 21+5
                box_w, box_h = 45, 35
                rects=[]
                for r in range(rows):
                    for c in range(cols):
                        if len(rects)>=17: break
                        x = int(offset_x + c*(box_w+pad_x))
                        y = int(offset_y + r*(box_h+pad_y))
                        rects.append((x,y,x+box_w,y+box_h))

                # Median-Lab
                medians=[]
                for r in rects:
                    med = median_color_in_rect(warped,r)
                    medians.append(opencv_lab_to_cie_lab(med))

                # Sample- und Referenz-ROIs
                even_rois = medians[3::2]
                odd_rois = medians[2::2]
                y_L = np.array([lab[0] for lab in even_rois])
                y_a = np.array([lab[1] for lab in even_rois])
                y_b = np.array([lab[2] for lab in even_rois])
                y_L_ref = np.median([lab[0] for lab in odd_rois])
                y_a_ref = np.median([lab[1] for lab in odd_rois])
                y_b_ref = np.median([lab[2] for lab in odd_rois])

                # Linear Fit 6.8–8.2
                mask = (self.ref_pH>=6.8)&(self.ref_pH<=8.2)
                x_lin = self.ref_pH[mask].reshape(-1,1)
                y_L_lin, y_a_lin, y_b_lin = y_L[mask], y_a[mask], y_b[mask]
                model_L = LinearRegression().fit(x_lin,y_L_lin)
                model_a = LinearRegression().fit(x_lin,y_a_lin)
                model_b = LinearRegression().fit(x_lin,y_b_lin)
                x_fine = np.linspace(self.ref_pH.min(),self.ref_pH.max(),1000)
                y_L_fit = model_L.predict(x_fine.reshape(-1,1))
                y_a_fit = model_a.predict(x_fine.reshape(-1,1))
                y_b_fit = model_b.predict(x_fine.reshape(-1,1))

                errors = [delta_e_ciede2000((L,a,b),(y_L_ref,y_a_ref,y_b_ref))
                          for L,a,b in zip(y_L_fit,y_a_fit,y_b_fit)]
                best_idx = np.argmin(errors)
                estimated_pH = x_fine[best_idx]

                # ---------------- Annotiertes Bild ----------------
                annotated = warped.copy()
                for i,(r,lab) in enumerate(zip(rects,medians)):
                    x1,y1,x2,y2 = r
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                    lab_bgr = cv2.cvtColor(np.uint8([[[lab[0]*255/100,lab[1]+128,lab[2]+128]]]),cv2.COLOR_Lab2BGR)[0,0]
                    cv2.rectangle(annotated,(x2+2,y1),(x2+2+20,y1+20),tuple(int(v) for v in lab_bgr),-1)
                    cv2.putText(annotated,str(i+1),(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

                self.label.text=f'Indikator erkannt! Geschätzter pH: {estimated_pH:.2f}'
                frame = annotated

        buf = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.img_widget.texture = texture

class MyApp(App):
    def build(self):
        return PHApp()

if __name__ == '__main__':
    MyApp().run()
