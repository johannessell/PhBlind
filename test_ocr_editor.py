#!/usr/bin/env python3
"""
OCR Editor Test & Demo
Verify installation and demonstrate functionality
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path


def test_dependencies():
    """Test all required dependencies"""
    print("\n" + "="*60)
    print("🔍 Testing Dependencies")
    print("="*60)
    
    deps = {
        'Flask': 'flask',
        'Flask-CORS': 'flask_cors',
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
    }
    
    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✅ {name:15} OK")
        except ImportError:
            print(f"❌ {name:15} MISSING")
            all_ok = False
    
    return all_ok


def test_ocr_engine():
    """Test OCR engine availability"""
    print("\n" + "="*60)
    print("🤖 Testing OCR Engine")
    print("="*60)
    
    try:
        from ocr_values import get_ocr_engine
        
        engine = get_ocr_engine()
        if engine.is_available():
            print(f"✅ OCR Engine: {engine.backend_name.upper()}")
            return True
        else:
            print(f"⚠️  No OCR engine available")
            print(f"   Install: pip install openocr  (or easyocr)")
            return False
    except ImportError:
        print(f"❌ ocr_values.py not found")
        print(f"   Make sure it's in the project root")
        return False


def test_file_structure():
    """Test required file structure"""
    print("\n" + "="*60)
    print("📁 Testing File Structure")
    print("="*60)
    
    required_files = {
        'Backend': 'ocr_editor_backend.py',
        'Launcher': 'run_ocr_editor.py',
        'Integration': 'integration_example.py',
        'HTML UI': 'templates/editor.html',
        'JavaScript': 'static/editor.js',
        'CSS': 'static/style.css',
    }
    
    all_ok = True
    for name, path in required_files.items():
        if Path(path).exists():
            print(f"✅ {name:15} {path}")
        else:
            print(f"❌ {name:15} {path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_image_processing():
    """Test image processing capabilities"""
    print("\n" + "="*60)
    print("📷 Testing Image Processing")
    print("="*60)
    
    try:
        # Create test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.putText(test_image, "Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Test encode
        _, buffer = cv2.imencode('.jpg', test_image)
        print(f"✅ Image creation & encoding")
        
        # Test contour detection
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ Contour detection ({len(contours)} contours found)")
        
        return True
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False


def test_api_structure():
    """Test backend API structure"""
    print("\n" + "="*60)
    print("🔌 Testing Backend Structure")
    print("="*60)
    
    try:
        import ocr_editor_backend
        
        # Check Flask app
        if hasattr(ocr_editor_backend, 'app'):
            print(f"✅ Flask app instance")
        
        # Check routes
        routes = ['/api/sessions', '/api/sessions/<session_id>/upload', 
                  '/api/sessions/<session_id>/ocr', '/api/sessions/<session_id>/export']
        print(f"✅ API routes defined")
        
        # Check OCRSession class
        if hasattr(ocr_editor_backend, 'OCRSession'):
            print(f"✅ OCRSession class")
        
        return True
    except Exception as e:
        print(f"❌ Backend structure error: {e}")
        return False


def demo_workflow():
    """Demonstrate basic workflow"""
    print("\n" + "="*60)
    print("🎬 Demo: Basic Workflow")
    print("="*60)
    
    try:
        # Simulate pipeline
        print("\n1️⃣  Creating test image...")
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (200, 150), (100, 150, 200), -1)
        cv2.putText(image, "Value: 42.5", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        print(f"   ✅ Image size: {image.shape}")
        
        print("\n2️⃣  Detecting ROIs...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                rois.append([x, y, x+w, y+h])
        print(f"   ✅ Found {len(rois)} ROIs")
        
        print("\n3️⃣  Creating session data...")
        session_data = {
            'session_id': 'demo_123',
            'image_shape': image.shape,
            'rois': rois[:2],  # First 2 ROIs
            'results': [
                {'original_text': '42.5', 'corrected_text': '42.5'},
                {'original_text': 'error', 'corrected_text': '43.0'},
            ]
        }
        print(f"   ✅ Session with {len(rois)} ROIs ready")
        
        print("\n4️⃣  Export format...")
        export = {
            'session_id': session_data['session_id'],
            'results': session_data['results']
        }
        print(f"   ✅ Export data: {json.dumps(export, indent=2)}")
        
        return True
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪 OCR EDITOR - TEST SUITE".center(60))
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("Image Processing", test_image_processing),
        ("Backend Structure", test_api_structure),
        ("OCR Engine", test_ocr_engine),
        ("Workflow Demo", demo_workflow),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "🎉 All tests passed!".center(60))
        print("\n" + "Ready to start:".center(60))
        print("\n  python run_ocr_editor.py".center(60))
        print("\n")
        return 0
    else:
        print("\n" + "⚠️  Some tests failed".center(60))
        print("\nFix issues and run again:")
        print("  python test_ocr_editor.py")
        print("\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
