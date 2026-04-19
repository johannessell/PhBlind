"""
Integration with contour_detection03.py - Direct OCR Import
===========================================================

This shows how to export OCR results from contour_detection03 and 
import them directly into the OCR Editor for user correction.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from openocr import OpenOCR
import requests


def extract_ocr_blocks_from_contours(image_path, output_json=None):
    """
    Replicate contour_detection03.py logic to extract OCR blocks.
    
    This is a simplified version - use the actual contour_detection03 module for production.
    
    Args:
        image_path: Path to image file
        output_json: Optional path to save OCR blocks as JSON
    
    Returns:
        List of OCR blocks: [{'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float}, ...]
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocessing (same as contour_detection03)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    canny = cv2.Canny(gray_enhanced, 30, 120)
    
    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 10:
                rects.append((x, y, w, h))
    
    # Get bounding box
    if not rects:
        raise ValueError("No rectangles found in image")
    
    margin = 50
    x_min = max(min(r[0] for r in rects) - margin, 0)
    y_min = max(min(r[1] for r in rects) - margin, 0)
    x_max = min(max(r[0] + r[2] for r in rects) + margin, image.shape[1])
    y_max = min(max(r[1] + r[3] for r in rects) + margin, image.shape[0])
    
    # Crop image
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    
    # OCR processing
    ocr = OpenOCR(backend='onnx', device='cpu')
    ocr_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Save temp file for OCR (use proper Windows/Linux temp path)
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, ocr_image)
    
    try:
        result, _ = ocr(temp_path)
    except Exception as e:
        print(f"Warning: OCR processing failed: {e}")
        result = []
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    # Parse OCR results
    ocr_blocks = parse_ocr_result(result)
    
    # Adjust coordinates to original image space
    for block in ocr_blocks:
        block['x'] += x_min
        block['y'] += y_min
    
    # Save if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(ocr_blocks, f, indent=2)
        print(f"✅ OCR blocks saved to {output_json}")
    
    return ocr_blocks


def parse_ocr_result(result):
    """
    Parse OpenOCR output to extract text blocks.
    
    Args:
        result: OpenOCR result
    
    Returns:
        List of blocks: [{'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float}, ...]
    """
    blocks = []
    
    if not result:
        return blocks
    
    for item in result:
        if not item:
            continue
        
        # Handle different result formats
        if isinstance(item, str):
            json_str = item.split('\t')[-1].strip()
        else:
            json_str = item
        
        try:
            detections = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            continue
        
        if not isinstance(detections, list):
            detections = [detections]
        
        for det in detections:
            text = det.get('transcription', '').strip()
            score = det.get('score', 0.0)
            pts = det.get('points', [])
            
            if not text or not pts:
                continue
            
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            
            blocks.append({
                'text': text,
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'score': float(score)
            })
    
    return blocks


# ============================================================================
# OCR Editor Integration
# ============================================================================

class OCREditorClient:
    """Client for interacting with OCR Editor via REST API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
    
    def create_session(self):
        """Create new editor session"""
        try:
            res = requests.post(f"{self.base_url}/api/sessions")
            data = res.json()
            self.session_id = data['session_id']
            return self.session_id
        except Exception as e:
            print(f"❌ Failed to create session: {e}")
            return None
    
    def upload_image(self, image_path):
        """Upload image to session"""
        if not self.session_id:
            raise ValueError("No session created")
        
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/jpeg')}
            res = requests.post(
                f"{self.base_url}/api/sessions/{self.session_id}/upload",
                files=files
            )
        
        if res.status_code != 200:
            raise Exception(f"Upload failed: {res.text}")
        
        return res.json()
    
    def import_ocr_blocks(self, ocr_blocks):
        """Import OCR blocks for editing"""
        if not self.session_id:
            raise ValueError("No session created")
        
        res = requests.post(
            f"{self.base_url}/api/sessions/{self.session_id}/ocr-import",
            json={'ocr_blocks': ocr_blocks}
        )
        
        if res.status_code != 200:
            raise Exception(f"Import failed: {res.text}")
        
        return res.json()
    
    def export_results(self):
        """Export corrected results"""
        if not self.session_id:
            raise ValueError("No session created")
        
        res = requests.get(
            f"{self.base_url}/api/sessions/{self.session_id}/export"
        )
        
        if res.status_code != 200:
            raise Exception(f"Export failed: {res.text}")
        
        return res.json()
    
    def open_editor(self):
        """Open editor in browser with session ID in URL"""
        import webbrowser
        url = f"{self.base_url}/?session={self.session_id}"
        print(f"🌐 Opening {url}...")
        webbrowser.open(url)


# ============================================================================
# Examples
# ============================================================================

def example_from_file():
    """Example: Load image, extract OCR with contours, import to editor"""
    print("\n" + "="*60)
    print("Example: Import OCR from contour_detection03")
    print("="*60)
    
    # Try different image paths
    image_paths = [
        "reference2.jpg",
        "img/PXL_20250808_175147650.MP.jpg_warp",
        "img/PXL_20250808_175154582.MP.jpg_warp",
    ]
    
    image_path = None
    for path in image_paths:
        if Path(path).exists():
            image_path = path
            break
    
    if not image_path:
        print(f"❌ Could not find image. Tried: {image_paths}")
        print(f"   Available files:")
        for p in Path('img').glob('*')[:3]:
            print(f"     - {p}")
        return
    
    try:
        # Step 1: Extract OCR blocks using contour detection
        print(f"\n📷 Using image: {image_path}")
        print(f"📍 Extracting OCR blocks from contours...")
        ocr_blocks = extract_ocr_blocks_from_contours(image_path)
        print(f"✅ Extracted {len(ocr_blocks)} text blocks")
        
        if not ocr_blocks:
            print("⚠️  No text blocks found - check image file")
            return
        
        for i, block in enumerate(ocr_blocks[:5]):
            print(f"   Block {i+1}: '{block['text']}' @ ({block['x']}, {block['y']})")
        
        # Step 2: Create OCR Editor session
        print("\n🔗 Connecting to OCR Editor...")
        client = OCREditorClient("http://localhost:5000")
        session_id = client.create_session()
        
        if not session_id:
            print("❌ Could not create session - is OCR Editor running?")
            print("   Start it with: python run_ocr_editor.py")
            return
        
        print(f"✅ Session created: {session_id}")
        
        # Step 3: Upload image
        print(f"\n📤 Uploading image...")
        client.upload_image(image_path)
        print(f"✅ Image uploaded")
        
        # Step 4: Import OCR blocks
        print(f"\n📥 Importing OCR blocks...")
        results = client.import_ocr_blocks(ocr_blocks)
        print(f"✅ {len(results['results'])} blocks ready for editing")
        
        # Step 5: Open editor
        print(f"\n👤 Opening editor for user correction...")
        client.open_editor()
        
        print(f"\n💾 After editing, use:")
        print(f"   results = client.export_results()")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_batch_with_json():
    """Example: Process multiple images and save OCR blocks"""
    print("\n" + "="*60)
    print("Example: Batch Process & Save OCR Blocks")
    print("="*60)
    
    image_dir = Path('img')
    output_dir = Path('ocr_blocks')
    output_dir.mkdir(exist_ok=True)
    
    image_files = list(image_dir.glob('*.jpg'))[:2]
    
    for image_path in image_files:
        print(f"\n📷 Processing {image_path.name}...")
        
        try:
            output_json = output_dir / f"{image_path.stem}_ocr.json"
            ocr_blocks = extract_ocr_blocks_from_contours(str(image_path), str(output_json))
            print(f"✅ Extracted {len(ocr_blocks)} blocks")
        
        except Exception as e:
            print(f"⚠️  Skipped: {e}")
    
    print(f"\n💾 OCR blocks saved to {output_dir}/")


if __name__ == '__main__':
    import sys
    
    print("\n🔍 OCR Editor - contour_detection03 Integration\n")
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == '1':
            example_from_file()
        elif example == '2':
            example_batch_with_json()
    else:
        print("Usage:")
        print("  python integration_ocr_import.py 1  - Import from contours & edit")
        print("  python integration_ocr_import.py 2  - Batch process & save blocks")
