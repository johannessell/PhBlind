"""
OCR Result Editor Backend
=========================
Flask-based REST API for interactive OCR result editing.
Mobile-friendly and desktop compatible.

Features:
- Upload/process images with OCR
- Edit recognized values
- Adjust ROI positions interactively
- Save corrected results
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import json
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import your existing OCR module
try:
    from ocr_values import get_ocr_engine, preprocess_for_ocr
except ImportError:
    print("Warning: ocr_values module not found. Using fallback OCR.")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('ocr_sessions')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Session storage
sessions: Dict[str, Any] = {}


class OCRSession:
    """Manages a single OCR editing session"""
    
    def __init__(self, session_id: str):
        self.id = session_id
        self.image = None
        self.image_display = None  # Resized for display
        self.ocr_results = []  # List of {roi, text, confidence, corrected_text}
        self.rois = []  # List of [x1, y1, x2, y2]
        self.metadata = {
            'created': datetime.now().isoformat(),
            'original_filename': None,
            'image_shape': None,
        }
    
    def save_session(self):
        """Save session to JSON file"""
        session_file = Path(app.config['UPLOAD_FOLDER']) / f"{self.id}_session.json"
        data = {
            'metadata': self.metadata,
            'ocr_results': [
                {
                    'roi': roi,
                    'text': result['text'],
                    'confidence': result.get('confidence', 0),
                    'corrected_text': result.get('corrected_text', None),
                }
                for roi, result in zip(self.rois, self.ocr_results)
            ]
        }
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
        return str(session_file)
    
    def to_dict(self):
        """Convert session to API response format"""
        return {
            'id': self.id,
            'metadata': self.metadata,
            'results': [
                {
                    'id': i,
                    'roi': roi,
                    'text': result['text'],
                    'confidence': result.get('confidence', 0),
                    'corrected_text': result.get('corrected_text', result['text']),
                    'is_corrected': result.get('corrected_text') is not None and 
                                   result.get('corrected_text') != result['text'],
                }
                for i, (roi, result) in enumerate(zip(self.rois, self.ocr_results))
            ]
        }


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main editor page"""
    return render_template('editor.html')


@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create new editing session"""
    session_id = f"session_{int(datetime.now().timestamp())}"
    session = OCRSession(session_id)
    sessions[session_id] = session
    return jsonify({'session_id': session_id})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session data"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    return jsonify(session.to_dict())


@app.route('/api/sessions/<session_id>/upload', methods=['POST'])
def upload_image(session_id):
    """Upload and process image with OCR"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        file_data = file.read()
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        session.image = image
        session.metadata['original_filename'] = file.filename
        session.metadata['image_shape'] = image.shape
        session.metadata['original_width'] = image.shape[1]  # Store original width
        session.metadata['original_height'] = image.shape[0]  # Store original height
        
        # Create display version (max 1280px width)
        display = image.copy()
        h, w = display.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display = cv2.resize(display, (1280, int(h * scale)))
        session.image_display = display
        
        print(f"[DEBUG] Image uploaded for session {session_id}, display shape: {session.image_display.shape}")
        
        return jsonify({
            'status': 'success',
            'image_shape': {
                'height': image.shape[0],
                'width': image.shape[1],
            },
            'display_shape': {
                'height': session.image_display.shape[0],
                'width': session.image_display.shape[1],
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sessions/<session_id>/image', methods=['GET'])
def get_image(session_id):
    """Get current session image as base64"""
    print(f"[DEBUG] get_image called for session: {session_id}")
    print(f"[DEBUG] Sessions available: {list(sessions.keys())}")
    
    if session_id not in sessions:
        print(f"[DEBUG] Session not found: {session_id}")
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    print(f"[DEBUG] session.image_display is None: {session.image_display is None}")
    
    if session.image_display is None:
        print(f"[DEBUG] No image loaded for session {session_id}")
        return jsonify({'error': 'No image loaded'}), 400
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', session.image_display)
    img_base64 = base64.b64encode(buffer).decode()
    
    print(f"[DEBUG] Returning image, base64 length: {len(img_base64)}")
    return jsonify({'image': f"data:image/jpeg;base64,{img_base64}"})


@app.route('/api/sessions/<session_id>/ocr', methods=['POST'])
def process_ocr(session_id):
    """Process image with OCR on defined ROIs"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    if session.image is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    try:
        data = request.get_json()
        rois = data.get('rois', [])  # List of [x1, y1, x2, y2]
        
        if not rois:
            return jsonify({'error': 'No ROIs provided'}), 400
        
        session.rois = rois
        session.ocr_results = []
        
        # Get OCR engine
        ocr_engine = get_ocr_engine('openocr')
        
        for roi in rois:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1, x2 = max(0, x1), min(session.image.shape[1], x2)
            y1, y2 = max(0, y1), min(session.image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                session.ocr_results.append({
                    'text': '',
                    'confidence': 0,
                })
                continue
            
            # Extract ROI
            roi_image = session.image[y1:y2, x1:x2]
            
            # Preprocess for OCR
            roi_prep = preprocess_for_ocr(roi_image, scale=2.0)
            
            # Read text
            text = ocr_engine.read_image(roi_prep) if ocr_engine.is_available() else ""
            
            session.ocr_results.append({
                'text': text.strip(),
                'confidence': 0.8,  # Placeholder
            })
        
        return jsonify(session.to_dict())
    
    except Exception as e:
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 400


@app.route('/api/sessions/<session_id>/ocr-auto', methods=['POST'])
def process_ocr_auto(session_id):
    """Process image with auto-detected ROIs from contour detection"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    if session.image is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    try:
        # Auto-detect ROIs using contour detection
        gray = cv2.cvtColor(session.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 200, 200)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        edges = cv2.Canny(gray_enhanced, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 50000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                rois.append([x, y, x + w, y + h])
        
        # Sort by position (top-to-bottom, left-to-right)
        rois = sorted(rois, key=lambda r: (r[1], r[0]))
        
        # Process OCR on detected ROIs
        session.rois = rois
        session.ocr_results = []
        ocr_engine = get_ocr_engine('openocr')
        
        for roi in rois:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1, x2 = max(0, x1), min(session.image.shape[1], x2)
            y1, y2 = max(0, y1), min(session.image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                session.ocr_results.append({'text': '', 'confidence': 0})
                continue
            
            roi_image = session.image[y1:y2, x1:x2]
            roi_prep = preprocess_for_ocr(roi_image, scale=2.0)
            text = ocr_engine.read_image(roi_prep) if ocr_engine.is_available() else ""
            
            session.ocr_results.append({'text': text.strip(), 'confidence': 0.8})
        
        return jsonify(session.to_dict())
    
    except Exception as e:
        return jsonify({'error': f'Auto-detection failed: {str(e)}'}), 400


@app.route('/api/sessions/<session_id>/ocr-import', methods=['POST'])
def import_ocr_results(session_id):
    """Import pre-computed OCR results from contour_detection03.py"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    if session.image is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    try:
        data = request.get_json()
        ocr_blocks = data.get('ocr_blocks', [])  # List of {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float}
        
        if not ocr_blocks:
            return jsonify({'error': 'No OCR blocks provided'}), 400
        
        # Convert OCR blocks to ROIs format
        session.rois = []
        session.ocr_results = []
        
        for block in ocr_blocks:
            x = block.get('x', 0)
            y = block.get('y', 0)
            w = block.get('w', 0)
            h = block.get('h', 0)
            text = block.get('text', '').strip()
            score = block.get('score', 0.8)
            
            roi = [x, y, x + w, y + h]
            session.rois.append(roi)
            session.ocr_results.append({
                'text': text,
                'confidence': float(score)
            })
        
        return jsonify(session.to_dict())
    
    except Exception as e:
        return jsonify({'error': f'Import failed: {str(e)}'}), 400


@app.route('/api/sessions/<session_id>/results/<int:result_id>', methods=['PATCH'])
def update_result(session_id, result_id):
    """Update corrected text for a result"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    if result_id >= len(session.ocr_results):
        return jsonify({'error': 'Result ID out of range'}), 404
    
    try:
        data = request.get_json()
        corrected_text = data.get('corrected_text', '').strip()
        
        session.ocr_results[result_id]['corrected_text'] = corrected_text
        
        return jsonify({
            'status': 'success',
            'result_id': result_id,
            'corrected_text': corrected_text,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sessions/<session_id>/results/<int:result_id>/roi', methods=['PATCH'])
def update_roi(session_id, result_id):
    """Update ROI coordinates for a result"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    if result_id >= len(session.rois):
        return jsonify({'error': 'Result ID out of range'}), 404
    
    try:
        data = request.get_json()
        roi = data.get('roi', [])  # [x1, y1, x2, y2]
        
        if len(roi) != 4:
            return jsonify({'error': 'Invalid ROI format'}), 400
        
        session.rois[result_id] = [int(v) for v in roi]
        
        return jsonify({
            'status': 'success',
            'result_id': result_id,
            'roi': session.rois[result_id],
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sessions/<session_id>/export', methods=['GET'])
def export_results(session_id):
    """Export corrected results"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    
    # Prepare export data
    export_data = {
        'session_id': session.id,
        'metadata': session.metadata,
        'results': [
            {
                'roi': roi,
                'original_text': result['text'],
                'corrected_text': result.get('corrected_text', result['text']),
                'confidence': result.get('confidence', 0),
            }
            for roi, result in zip(session.rois, session.ocr_results)
        ]
    }
    
    # Save session
    session.save_session()
    
    return jsonify(export_data)


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
