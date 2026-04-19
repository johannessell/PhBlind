# OCR Result Editor - Setup & Usage Guide

Interactive web-based GUI for editing OCR results with ROI adjustment. Works on desktop and mobile browsers.

## Features

✅ **Image Upload** - Drag & drop or select images  
✅ **ROI Definition** - Click & drag to define regions for text recognition  
✅ **OCR Processing** - Automatic text recognition from ROIs  
✅ **Interactive Editing** - Correct recognized values in-place  
✅ **ROI Adjustment** - Resize and reposition ROI regions  
✅ **Export Results** - Save corrected results as JSON  
✅ **Mobile Responsive** - Works on smartphones and tablets  

## Installation

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements_ocr_editor.txt
```

### 2. Verify Your OCR Setup

The editor integrates with your existing `ocr_values.py` module. Make sure you have:
- `openocr` or `easyocr` installed
- `ocr_values.py` in the project root

```bash
# If you don't have openocr installed:
pip install openocr

# Or use easyocr:
pip install easyocr
```

## Usage

### Start the Server

```bash
python ocr_editor_backend.py
```

The app will start on `http://localhost:5000`

### Access from Different Devices

**Same Computer:**
- Open browser → `http://localhost:5000`

**Smartphone on Same Network:**
- Get your computer's IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
- On phone, open: `http://<YOUR_IP>:5000`

**Remote Server:**
- Change `host` in `ocr_editor_backend.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5000)
  ```

## Workflow

### Step 1: Upload Image
1. Click upload area or drag & drop an image
2. Image is displayed on canvas

### Step 2: Define ROIs (Regions of Interest)
1. Click and drag on image to create rectangular regions
2. Each ROI will be processed for OCR
3. Use "Clear All ROIs" to start over
4. Click "Process OCR" when ready

### Step 3: Review & Correct
1. OCR results appear as cards
2. Edit values directly in text fields
3. Click "Edit" for modal editing
4. Corrected values are highlighted

### Step 4: Export
1. Click "Export Results"
2. JSON file is downloaded with:
   - Original OCR text
   - Corrected text
   - ROI coordinates
   - Metadata

## API Endpoints

```
POST   /api/sessions                     - Create new session
POST   /api/sessions/<id>/upload         - Upload image
GET    /api/sessions/<id>/image          - Get image as base64
POST   /api/sessions/<id>/ocr            - Process OCR on ROIs
PATCH  /api/sessions/<id>/results/<rid>  - Update corrected text
PATCH  /api/sessions/<id>/results/<rid>/roi - Update ROI coordinates
GET    /api/sessions/<id>/export         - Export results
DELETE /api/sessions/<id>                - Delete session
```

## Configuration

### Backend Settings

Edit `ocr_editor_backend.py`:

```python
# Max upload size
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Preferred OCR engine ('openocr', 'easyocr', 'tesseract')
ocr_engine = get_ocr_engine('openocr')
```

### Frontend Settings

Edit `static/editor.js`:

```javascript
// API host and port
const API_URL = window.location.origin;  // Auto-detects

// Canvas max width
const maxWidth = 800;
```

## Integration with Your Project

### Using Results in Your Code

```python
import json

# Load exported JSON
with open('ocr_results_session_*.json', 'r') as f:
    results = json.load(f)

# Extract corrected values
for result in results['results']:
    roi = result['roi']          # [x1, y1, x2, y2]
    original = result['original_text']
    corrected = result['corrected_text']
    
    # Use corrected values
    print(f"ROI {roi}: {corrected}")
```

### Integrate with Contour Detection

```python
from ocr_editor_backend import OCRSession

# After contour detection, create ROIs
rois = [
    [x1, y1, x2, y2],  # ROI 1
    [x3, y3, x4, y4],  # ROI 2
]

# Process through editor
session = OCRSession('manual_session')
session.image = contour_image
session.rois = rois
session.ocr_results = process_rois(contour_image, rois)
```

## Deployment

### Docker Container

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_ocr_editor.txt .
RUN pip install -r requirements_ocr_editor.txt

COPY . .

EXPOSE 5000
CMD ["python", "ocr_editor_backend.py"]
```

Build & run:
```bash
docker build -t ocr-editor .
docker run -p 5000:5000 ocr-editor
```

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 ocr_editor_backend:app
```

## Smartphone App Integration

### React Native / Flutter Integration

The editor exposes a REST API - you can create native mobile apps that call:

```javascript
// Upload image to editor
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://server:5000/api/sessions/<id>/upload', {
    method: 'POST',
    body: formData
});
```

### Web Wrapper for Mobile

Use tools like:
- **Electron** - Desktop app from web
- **Capacitor** - iOS/Android from web
- **Tauri** - Lightweight desktop wrapper

## Troubleshooting

### Port Already in Use

```bash
# Windows: Find process using port 5000
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F
```

### OCR Not Working

1. Verify `ocr_values.py` is in project root
2. Test OCR engine:
   ```bash
   python -c "from ocr_values import get_ocr_engine; e = get_ocr_engine(); print(e.is_available())"
   ```

### Image Not Uploading

- Check max file size: `app.config['MAX_CONTENT_LENGTH']`
- Verify browser console for errors (F12)
- Check server logs for error details

### CORS Issues

- Already configured with `flask-cors`
- For custom domains, modify CORS settings in backend

## File Structure

```
PhBlind/
├── ocr_editor_backend.py          # Flask backend
├── templates/
│   └── editor.html                # Web UI template
├── static/
│   ├── editor.js                  # Frontend logic
│   └── style.css                  # Styling
├── ocr_sessions/                  # Session storage
├── requirements_ocr_editor.txt    # Dependencies
└── OCR_EDITOR_README.md           # This file
```

## Next Steps

1. ✅ Start the backend server
2. ✅ Open `http://localhost:5000` in browser
3. ✅ Upload an image
4. ✅ Define ROIs for your OCR values
5. ✅ Review and correct recognized text
6. ✅ Export results as JSON
7. ✅ Integrate results back into your pipeline

---

For mobile deployment, see **Smartphone App Integration** section above.
