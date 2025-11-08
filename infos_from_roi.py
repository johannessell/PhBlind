import os
import cv2
import numpy as np
import pandas as pd
import argparse
import csv

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ------------------------------
# Color utilities
# ------------------------------

def median_color_in_rect(img_bgr, rect):
    """
    Compute median color in ROI and return Lab values (OpenCV format).
    rect: (x1,y1,x2,y2)
    """
    x1,y1,x2,y2 = rect
    x1, x2 = int(round(min(x1,x2))), int(round(max(x1,x2)))
    y1, y2 = int(round(min(y1,y2))), int(round(max(y1,y2)))
    h, w = img_bgr.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (0,0,0), (x1,y1,x2-x1,y2-y1)

    patch_bgr = img_bgr[y1:y2, x1:x2]
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2Lab)
    med = np.median(patch_lab.reshape(-1,3), axis=0)
    return (float(med[0]), float(med[1]), float(med[2])), (x1,y1,x2-x1,y2-y1)

def opencv_lab_to_cie_lab(lab_value):
    """
    Convert Lab from OpenCV's 8-bit representation to standard CIE Lab.
    """
    L, a, b = lab_value
    L_cie = L * 100.0 / 255.0
    a_cie = a - 128.0
    b_cie = b - 128.0
    return (L_cie, a_cie, b_cie)


def cie_lab_to_opencv_lab(cie_lab):
    L, a, b = cie_lab
    L_cv = int(round(L * 255 / 100))
    a_cv = int(round(a + 128))
    b_cv = int(round(b + 128))
    return np.uint8([[[L_cv, a_cv, b_cv]]])

# ------------------------------
# CIEDE2000 ΔE implementation
# ------------------------------

def delta_e_ciede2000(lab1, lab2):
    """Compute CIEDE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0

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

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    if abs(h1p - h2p) > 180:
        avg_hp = (h1p + h2p + 360) / 2.0
    else:
        avg_hp = (h1p + h2p) / 2.0

    T = (1 - 0.17*np.cos(np.radians(avg_hp - 30))
         + 0.24*np.cos(np.radians(2*avg_hp))
         + 0.32*np.cos(np.radians(3*avg_hp + 6))
         - 0.20*np.cos(np.radians(4*avg_hp - 63)))

    d_ro = 30 * np.exp(-((avg_hp - 275)/25)**2)
    RC = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    SL = 1 + (0.015 * (avg_Lp - 50)**2) / np.sqrt(20 + (avg_Lp - 50)**2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T
    RT = -np.sin(2 * np.radians(d_ro)) * RC

    dE = np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH))
    return dE

# ------------------------------
# ROI utilities
# ------------------------------

def draw_annotation(img_bgr, rects, medians_lab):
    img = img_bgr.copy()
    for i, (r, lab) in enumerate(zip(rects, medians_lab)):
        x, y, w, h = r
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        L, a, b = lab
        lab_pixel = cie_lab_to_opencv_lab((L, a, b))
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)[0, 0]
        sw = 30
        cv2.rectangle(img, (x + w + 6, y), (x + w + 6 + sw, y + sw),
                      tuple(int(v) for v in bgr_pixel), -1)
        cv2.putText(img, f"{i + 1}", (x, y - 6 if y - 6 > 10 else y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2, cv2.LINE_AA)
    return img

# ------------------------------
# Auto ROI placement
# ------------------------------

def auto_mode(img_path, n_rois=17):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img_bgr.shape[:2]
    cols, rows = 2, 8
    offset_x, offset_y = 95, 50+100+45
    pad_x, pad_y = 50, 21+5.5
    box_w, box_h = 45, 35
    rects = []
    for r in range(rows):
        for c in range(cols):
            if len(rects) >= n_rois:
                break
            x = int(offset_x + c*(box_w + pad_x))
            y = int(offset_y + r*(box_h + pad_y))
            x = max(0, min(w-1-box_w, x))
            y = max(0, min(h-1-box_h, y))
            rects.append((x,y,box_w,box_h))

    medians, rect_info = [], []
    for r in rects:
        x,y,w0,h0 = r
        med_lab, (xx,yy,ww,hh) = median_color_in_rect(img_bgr, (x,y,x+w0,y+h0))
        cie_lab = opencv_lab_to_cie_lab(med_lab)
        medians.append(cie_lab)
        rect_info.append((xx,yy,ww,hh))
    annotated = draw_annotation(img_bgr, rect_info, medians)
    return img_bgr, annotated, rect_info, medians

# ------------------------------
# Main analysis
# ------------------------------

def main():
    foldername = 'img/zoom/'
    filelist = [f for f in os.listdir(foldername) if f.endswith('.jpg')]

    for filename in filelist:
        file = os.path.join(foldername, filename)
        _, annotated, rects, medians = auto_mode(file, n_rois=17)

        cv2.imshow('annotated', annotated)

        # Example reference pH values (must match number of solution ROIs)
        x_values = np.array([8.2,7.8,7.6,7.4,7.2,7.0,6.8])

        # Split ROIs: even index = sample, odd index = reference
        even_rois = medians[3::2]
        odd_rois = medians[2::2]

        # Extract L*, a*, b* for samples and references
        y_L = np.array([lab[0] for lab in even_rois])
        y_a = np.array([lab[1] for lab in even_rois])
        y_b = np.array([lab[2] for lab in even_rois])

        y_L_ref = np.median([lab[0] for lab in odd_rois])
        y_a_ref = np.median([lab[1] for lab in odd_rois])
        y_b_ref = np.median([lab[2] for lab in odd_rois])

        # --------------------
        # Linear regression (restricted to 6.8–8.2)
        # --------------------
        mask = (x_values >= 6.8) & (x_values <= 8.2)
        x_lin = x_values[mask].reshape(-1, 1)
        y_L_lin, y_a_lin, y_b_lin = y_L[mask], y_a[mask], y_b[mask]

        model_L = LinearRegression().fit(x_lin, y_L_lin)
        model_a = LinearRegression().fit(x_lin, y_a_lin)
        model_b = LinearRegression().fit(x_lin, y_b_lin)

        x_fine = np.linspace(x_values.min(), x_values.max(), 1000).reshape(-1, 1)
        y_L_fit = model_L.predict(x_fine)
        y_a_fit = model_a.predict(x_fine)
        y_b_fit = model_b.predict(x_fine)

        errors_lin = [
            delta_e_ciede2000((L, aa, bb), (y_L_ref, y_a_ref, y_b_ref))
            for L, aa, bb in zip(y_L_fit, y_a_fit, y_b_fit)
        ]
        best_x_linear = x_fine[np.argmin(errors_lin)]
        print(f"{filename} → Best x (linear fit 6.8–8.2): {float(best_x_linear):.3f}")

        # --------------------
        # Cubic interpolation (for comparison)
        # --------------------
        f_L = interp1d(x_values, y_L, kind='cubic')
        f_a = interp1d(x_values, y_a, kind='cubic')
        f_b = interp1d(x_values, y_b, kind='cubic')

        y_L_interp = f_L(x_fine.flatten())
        y_a_interp = f_a(x_fine.flatten())
        y_b_interp = f_b(x_fine.flatten())

        errors_cub = [
            delta_e_ciede2000((L, aa, bb), (y_L_ref, y_a_ref, y_b_ref))
            for L, aa, bb in zip(y_L_interp, y_a_interp, y_b_interp)
        ]
        best_x_cubic = x_fine[np.argmin(errors_cub)]
        print(f"{filename} → Best x (cubic interpolation): {float(best_x_cubic):.3f}")

        # --------------------
        # Plotting
        # --------------------
        plt.figure(figsize=(8,5))
        plt.plot(x_values, y_a, 'ro-', label='a* samples')
        plt.plot(x_values, y_b, 'bo-', label='b* samples')
        plt.plot(x_values, y_L, 'go-', label='L* samples')

        plt.plot(x_fine, y_a_fit, 'r--', label='a* linear fit (6.8–8.2)')
        plt.plot(x_fine, y_b_fit, 'b--', label='b* linear fit (6.8–8.2)')
        plt.plot(x_fine, y_L_fit, 'g--', label='L* linear fit (6.8–8.2)')

        plt.axhline(y_a_ref, color='r', linestyle=':', label='a* ref median')
        plt.axhline(y_b_ref, color='b', linestyle=':', label='b* ref median')
        plt.axhline(y_L_ref, color='g', linestyle=':', label='L* ref median')

        plt.xlabel('pH')
        plt.ylabel('Lab components')
        plt.title(f"Color vs pH – {filename}")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
