import cv2, os, sys
sys.path.insert(0, 'android/PoolWaterTester/app/src/main/python')
import reference_builder as RB

expected = ['measure', 'color', 'color', 'measure',
            'color', 'color', 'measure']
for fn in sorted(os.listdir('reference_input/training')):
    if not fn.startswith('frame_17796'):
        continue
    bgr = cv2.imread(f'reference_input/training/{fn}')
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = RB._detect_reference_quad(gray, bgr)
    warped, _ = RB._warp_to_canonical(bgr, quad)
    rects = RB._detect_cells(warped)
    grid, sorted_cols, cp, cw, rp, rh = RB._build_grid(rects,
                                                      warped.shape[1])
    col_types, col_stats = RB._classify_columns(warped, grid)
    print(fn)
    for i in sorted(col_types):
        s = col_stats[i]['ab_spread']
        ok = 'OK' if col_types[i] == expected[i] else 'WRONG'
        print(f"  col {i}: spread={s:5.1f}  -> {col_types[i]:7s}"
              f"  (expected {expected[i]:7s}) {ok}")
    print()
