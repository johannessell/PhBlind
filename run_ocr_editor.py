#!/usr/bin/env python3
"""
OCR Editor Launcher
Quick start script for the OCR Result Editor
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("🔍  OCR Result Editor - Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    required = ['flask', 'flask_cors', 'cv2', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"   pip install -r requirements_ocr_editor.txt")
        return 1
    
    # Check OCR engine
    print("\n🤖 Checking OCR engine...")
    try:
        from ocr_values import get_ocr_engine
        engine = get_ocr_engine()
        if engine.is_available():
            print(f"   ✅ OCR Backend: {engine.backend_name}")
        else:
            print(f"   ⚠️  No OCR engine available")
            print(f"   Install: pip install openocr  (or easyocr, tesseract)")
    except ImportError:
        print(f"   ⚠️  ocr_values.py not found in current directory")
    
    # Create session folder
    sessions_dir = Path('ocr_sessions')
    sessions_dir.mkdir(exist_ok=True)
    
    # Start server
    print("\n🚀 Starting server...")
    print("   http://localhost:5000")
    print("\n   Press CTRL+C to stop\n")
    
    # Open browser after delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("   ✅ Browser opened\n")
        except:
            print("   💡 Open http://localhost:5000 in your browser\n")
    
    import threading
    thread = threading.Thread(target=open_browser, daemon=True)
    thread.start()
    
    # Run Flask app
    try:
        from ocr_editor_backend import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
