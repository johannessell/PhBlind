"""
Integration Example: Contour Detection + OCR Editor
====================================================

This example shows how to integrate your contour detection 
with the OCR Result Editor for smartphone apps.

Workflow:
1. Capture image from camera/file
2. Run contour detection to find ROI regions
3. Export ROIs to OCR Editor
4. User corrects OCR results via web GUI
5. Import corrected results back into your app
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import requests


class ContourOCRPipeline:
    """
    Integration layer between contour detection and OCR editor
    """
    
    def __init__(self, ocr_editor_url: str = "http://localhost:5000"):
        """
        Args:
            ocr_editor_url: Base URL of OCR editor (e.g., http://localhost:5000 or http://192.168.1.100:5000)
        """
        self.ocr_editor_url = ocr_editor_url.rstrip('/')
        self.session_id = None
        self.image = None
        self.rois = []
        self.results = {}
    
    # ========================================================================
    # Step 1: Contour Detection → Extract ROIs
    # ========================================================================
    
    def detect_rois_from_contours(self, image: np.ndarray, 
                                  min_area: int = 100,
                                  max_area: int = 10000) -> List[Tuple[int, int, int, int]]:
        """
        Extract ROIs from contour detection.
        
        This integrates with your contour_detection03.py logic.
        
        Args:
            image: Input image (BGR)
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
        
        Returns:
            List of ROIs: [(x1, y1, x2, y2), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter (like in contour_detection03.py)
        gray = cv2.bilateralFilter(gray, 5, 200, 200)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(gray_enhanced, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                rois.append((x, y, x + w, y + h))
        
        # Sort by position (top-to-bottom, left-to-right)
        rois = sorted(rois, key=lambda r: (r[1], r[0]))
        
        return rois
    
    # ========================================================================
    # Step 2: Send to OCR Editor
    # ========================================================================
    
    def upload_to_editor(self, image: np.ndarray, rois: List[Tuple[int, int, int, int]],
                        metadata: Dict = None) -> str:
        """
        Create session and upload image + ROIs to OCR Editor.
        
        Args:
            image: Input image (BGR)
            rois: List of ROI regions
            metadata: Optional metadata (filename, etc.)
        
        Returns:
            Session ID
        """
        self.image = image
        self.rois = rois
        
        try:
            # Create session
            print(f"📡 Connecting to {self.ocr_editor_url}...")
            session_res = requests.post(f"{self.ocr_editor_url}/api/sessions")
            session_data = session_res.json()
            self.session_id = session_data['session_id']
            print(f"✅ Session created: {self.session_id}")
            
            # Upload image
            _, buffer = cv2.imencode('.jpg', image)
            files = {'image': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
            
            upload_res = requests.post(
                f"{self.ocr_editor_url}/api/sessions/{self.session_id}/upload",
                files=files
            )
            
            if upload_res.status_code == 200:
                print(f"✅ Image uploaded ({image.shape[1]}x{image.shape[0]})")
            else:
                raise Exception(f"Upload failed: {upload_res.text}")
            
            # Process OCR
            ocr_res = requests.post(
                f"{self.ocr_editor_url}/api/sessions/{self.session_id}/ocr",
                json={'rois': rois}
            )
            
            if ocr_res.status_code == 200:
                self.results = ocr_res.json()
                print(f"✅ OCR processed {len(rois)} ROIs")
            else:
                raise Exception(f"OCR processing failed: {ocr_res.text}")
            
            return self.session_id
        
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to {self.ocr_editor_url}")
            print(f"   Start OCR editor: python run_ocr_editor.py")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def open_in_browser(self):
        """Open editor in web browser"""
        import webbrowser
        url = self.ocr_editor_url
        print(f"🌐 Opening {url}...")
        webbrowser.open(url)
    
    # ========================================================================
    # Step 3: Poll for Results (Optional - for offline usage)
    # ========================================================================
    
    def get_corrected_results(self, session_id: str = None) -> Dict:
        """
        Retrieve corrected results from OCR Editor.
        
        Args:
            session_id: Session ID (uses self.session_id if not provided)
        
        Returns:
            Results dictionary with corrected values
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID available")
        
        try:
            res = requests.get(f"{self.ocr_editor_url}/api/sessions/{sid}/export")
            return res.json()
        except Exception as e:
            print(f"❌ Failed to get results: {e}")
            return None
    
    # ========================================================================
    # Step 4: Use Corrected Results
    # ========================================================================
    
    def extract_values(self, results: Dict = None) -> Dict[int, float]:
        """
        Extract numeric values from corrected OCR results.
        
        Args:
            results: Results from get_corrected_results() or editor export
        
        Returns:
            Dictionary: {roi_index: float_value}
        """
        data = results or self.results
        values = {}
        
        for i, result in enumerate(data.get('results', [])):
            corrected_text = result.get('corrected_text', '').strip()
            
            # Try to parse as float
            try:
                value = float(corrected_text.replace(',', '.'))
                values[i] = value
            except ValueError:
                values[i] = None
        
        return values
    
    def save_results_to_file(self, filename: str, results: Dict = None):
        """
        Save corrected results to JSON file.
        
        Args:
            filename: Output filename
            results: Results data (uses self.results if not provided)
        """
        data = results or self.results
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved: {filename}")


# ============================================================================
# Example Usage
# ============================================================================

def example_desktop_workflow():
    """
    Example 1: Desktop workflow
    User captures image, defines ROIs, corrects via web GUI
    """
    print("\n" + "="*60)
    print("Example 1: Desktop Workflow")
    print("="*60)
    
    # Load image
    image = cv2.imread('reference2.jpg')  # Your image
    if image is None:
        print("❌ Image not found")
        return
    
    # Initialize pipeline
    pipeline = ContourOCRPipeline("http://localhost:5000")
    
    # Step 1: Detect ROIs from contours
    print("\n📍 Detecting ROIs from contours...")
    rois = pipeline.detect_rois_from_contours(image)
    print(f"✅ Found {len(rois)} ROIs")
    
    # Step 2: Upload to editor
    print("\n📤 Uploading to OCR Editor...")
    session_id = pipeline.upload_to_editor(image, rois)
    
    if session_id:
        # Step 3: Open in browser for user correction
        pipeline.open_in_browser()
        
        print("\n👤 User is correcting values in browser...")
        print("   Waiting 30 seconds before retrieving results...")
        
        import time
        time.sleep(30)
        
        # Step 4: Get corrected results
        print("\n📥 Retrieving corrected results...")
        results = pipeline.get_corrected_results()
        
        # Step 5: Extract values
        values = pipeline.extract_values(results)
        
        print("\n✅ Corrected Values:")
        for roi_idx, value in values.items():
            if value is not None:
                print(f"   ROI {roi_idx + 1}: {value}")
        
        # Step 6: Save
        pipeline.save_results_to_file('corrected_results.json', results)


def example_smartphone_workflow():
    """
    Example 2: Smartphone workflow
    For integration with mobile apps (React Native, Flutter, etc.)
    """
    print("\n" + "="*60)
    print("Example 2: Smartphone Workflow (API Integration)")
    print("="*60)
    
    # URL for remote server (e.g., your laptop or cloud server)
    remote_url = "http://192.168.1.100:5000"  # Change to your IP
    
    pipeline = ContourOCRPipeline(remote_url)
    
    # This would be called from mobile app:
    # 1. Mobile app captures image
    image = cv2.imread('reference2.jpg')  # Simulating mobile capture
    
    # 2. Mobile app detects ROIs locally or sends to backend
    rois = pipeline.detect_rois_from_contours(image)
    
    # 3. Mobile app sends to OCR editor
    session_id = pipeline.upload_to_editor(image, rois)
    
    # 4. User corrects on web GUI (or native mobile UI)
    print(f"\n📱 Share this link with user:")
    print(f"   {remote_url}")
    
    # 5. Mobile app polls for results
    # (In real app, use WebSocket or long-polling)
    import time
    time.sleep(5)
    
    results = pipeline.get_corrected_results()
    values = pipeline.extract_values(results)
    
    return values


def example_batch_processing():
    """
    Example 3: Batch processing
    Process multiple images and correct all at once
    """
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    pipeline = ContourOCRPipeline("http://localhost:5000")
    
    image_files = list(Path('img').glob('*.jpg'))
    all_results = {}
    
    for img_path in image_files[:3]:  # Process first 3
        print(f"\n📷 Processing {img_path.name}...")
        
        image = cv2.imread(str(img_path))
        rois = pipeline.detect_rois_from_contours(image)
        
        session_id = pipeline.upload_to_editor(image, rois, {'source': str(img_path)})
        
        if session_id:
            all_results[img_path.name] = {
                'session_id': session_id,
                'rois': rois
            }
    
    # Save session references
    with open('batch_sessions.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Batch session saved: batch_sessions.json")


if __name__ == '__main__':
    import sys
    
    print("\n🔍 Contour Detection + OCR Editor Integration Examples\n")
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == '1':
            example_desktop_workflow()
        elif example == '2':
            example_smartphone_workflow()
        elif example == '3':
            example_batch_processing()
    else:
        print("Usage:")
        print("  python integration_example.py 1  - Desktop workflow")
        print("  python integration_example.py 2  - Smartphone workflow")
        print("  python integration_example.py 3  - Batch processing")
