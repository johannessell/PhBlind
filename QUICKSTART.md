# OCR Editor - Quick Start

## 🚀 Start in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements_ocr_editor.txt
```

### 2. Run the Editor
```bash
python run_ocr_editor.py
```

### 3. Open Browser
- Automatic: Browser opens at `http://localhost:5000`
- Manual: Visit `http://localhost:5000`

---

## 📱 Usage Workflow

### **Option A: Auto-Detect ROIs (Recommended)**
1. Upload image
2. Click "Auto-Detect & Process OCR"
3. Editor automatically detects text regions
4. Review and correct values

### **Option B: Use OCR from contour_detection03.py**
This is the **best option** - uses your well-tested contour detection:

```python
from integration_ocr_import import OCREditorClient, extract_ocr_blocks_from_contours

# Extract OCR blocks using contour detection
ocr_blocks = extract_ocr_blocks_from_contours('reference2.jpg')

# Import to editor
client = OCREditorClient("http://localhost:5000")
client.create_session()
client.upload_image('reference2.jpg')
client.import_ocr_blocks(ocr_blocks)
client.open_editor()  # Opens browser for user correction
```

### **Option C: Manual ROI Drawing**
1. Upload image
2. Switch to "Manual ROI Drawing" mode
3. Click & drag to define text regions
4. Click "Process OCR"
5. Review and correct

---

## 📊 Result Format

After user editing, get corrected values:

```python
results = client.export_results()
# Returns:
# {
#   'results': [
#     {
#       'roi': [x1, y1, x2, y2],
#       'original_text': 'recognized text',
#       'corrected_text': 'user corrected text'
#     },
#     ...
#   ]
# }
```

---

## 🔄 Full Workflow Example

```python
from integration_ocr_import import OCREditorClient, extract_ocr_blocks_from_contours

# 1. Extract from your image
ocr_blocks = extract_ocr_blocks_from_contours('reference2.jpg', 'ocr_blocks.json')

# 2. Create editor session and upload
client = OCREditorClient("http://localhost:5000")
client.create_session()
client.upload_image('reference2.jpg')

# 3. Import OCR results
results = client.import_ocr_blocks(ocr_blocks)

# 4. Open for user correction
client.open_editor()

# 5. After user corrects (manual step), export
corrected = client.export_results()

# 6. Use corrected values
for item in corrected['results']:
    print(f"{item['original_text']} → {item['corrected_text']}")
```

---

## 🌐 Access from Smartphone

1. Find your computer's IP:
   - Windows: `ipconfig` → IPv4 address
   - Mac/Linux: `ifconfig` | grep inet

2. On phone, open: `http://<YOUR_IP>:5000`

3. Same workflow as desktop

---

## 📚 Integration with Your Code

### Direct Python Usage

```python
import json

# Load corrected results
with open('ocr_blocks.json') as f:
    ocr_blocks = json.load(f)

# Process through editor
client.import_ocr_blocks(ocr_blocks)

# Get corrected values
results = client.export_results()
corrected_values = {i: r['corrected_text'] for i, r in enumerate(results['results'])}
```

---

## ✨ Features

✅ **Auto-detect ROIs** from contours  
✅ **Import OCR results** from contour_detection03  
✅ **Manual ROI drawing** as fallback  
✅ **Interactive editing** of values  
✅ **Mobile responsive**  
✅ **Export corrected results**  

---

## 📁 Files Created

```
PhBlind/
├── 📄 ocr_editor_backend.py         ← Flask backend
├── 📄 run_ocr_editor.py             ← Launcher
├── 📄 integration_ocr_import.py     ← Import from contours ⭐
├── 📄 integration_example.py        ← Other integrations
├── templates/
│   └── editor.html                  ← Web UI
├── static/
│   ├── editor.js                    ← Frontend logic
│   └── style.css                    ← Styling
└── ocr_sessions/                    ← Session storage
```

---

## 🆘 Troubleshooting

**Editor not starting?**
```bash
python test_ocr_editor.py
```

**OCR results not importing?**
- Check `ocr_blocks` format (list of dicts with keys: text, x, y, w, h, score)
- Verify editor session is running

**Port already in use?**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

**Next: Use `integration_ocr_import.py 1` to try the full workflow!**

