import cv2, sys, os, glob
sys.path.insert(0, 'android/PoolWaterTester/app/src/main/python')
import measurement as M
import reference_builder as RB
M.init('')

FN = 'reference_input/training/frame_1780070551005.jpg'
OUT = 'new_frame_debug/frame_1780070551005'
os.makedirs(OUT, exist_ok=True)
RB.set_debug_dir(OUT)

bgr = cv2.imread(FN)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
quad = RB._detect_reference_quad(gray, bgr)
RB.set_debug_dir('')
warped = M._TRACKER.warp(bgr, quad)
cv2.imwrite(f'{OUT}/30_warped.jpg', warped)

grid, (rows, cols), allc = M._detect_warped_grid(warped)
vis = warped.copy()
ref_cells = {(c['row_idx'], c['col_idx']): c for c in M._REF['cells']}
for (r, cc), s in sorted(grid.items()):
    cx, cy, w, h = s[0], s[1], s[2], s[3]
    x, y = int(cx - w / 2), int(cy - h / 2)
    est = len(s) > 6 and s[6] == 'est'
    color = (0, 140, 255) if est else (0, 220, 0)
    cv2.rectangle(vis, (x, y), (x + int(w), y + int(h)), color, 2)
    refc = ref_cells.get((r, cc))
    lbl = f'{r},{cc}'
    if refc and refc.get('value') is not None:
        lbl += ':' + str(refc['value'])
    cv2.putText(vis, lbl, (x + 2, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
cv2.imwrite(f'{OUT}/31_runtime_grid.jpg', vis)
print('rows x cols:', rows, cols)
for f in sorted(glob.glob(f'{OUT}/*.jpg')):
    print(' ', os.path.basename(f))
