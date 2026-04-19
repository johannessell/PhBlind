/**
 * OCR Editor Frontend
 * Interactive UI for OCR result correction and ROI adjustment
 */

class OCREditor {
    constructor() {
        this.sessionId = null;
        this.currentImage = null;
        this.rois = [];
        this.results = [];
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentROI = null;
        this.scale = 1;
        this.editingIndex = null;

        this.initElements();
        this.attachEventListeners();
        
        // Check if session ID is provided in URL
        this.checkUrlForSession();
    }
    
    checkUrlForSession() {
        // Parse session ID from URL query string
        const params = new URLSearchParams(window.location.search);
        const urlSessionId = params.get('session');
        
        if (urlSessionId) {
            console.log('Loading session from URL:', urlSessionId);
            this.sessionId = urlSessionId;
            this.loadExistingSession();
        }
    }
    
    async loadExistingSession() {
        // Load the session state from backend
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}`);
            if (response.ok) {
                const data = await response.json();
                console.log('Session loaded:', data);
                
                // Set the results and skip to review step
                this.results = data.results || [];
                
                // Store original image dimensions for ROI scaling
                this.originalImageWidth = data.metadata?.original_width || 0;
                this.originalImageHeight = data.metadata?.original_height || 0;
                
                // Show step 3 directly
                this.moveToStep('review');
                this.displayResults();
            } else {
                console.error('Failed to load session');
                this.sessionId = null;
            }
        } catch (error) {
            console.error('Error loading session:', error);
            this.sessionId = null;
        }
    }

    initElements() {
        // File input
        this.fileInput = document.getElementById('imageInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.uploadArea = document.querySelector('.upload-area');

        // Mode selector
        this.autoDetectModeBtn = document.getElementById('autoDetectModeBtn');
        this.manualModeBtn = document.getElementById('manualModeBtn');
        this.autoDetectMode = document.getElementById('autoDetectMode');
        this.manualMode = document.getElementById('manualMode');

        // Canvas - for manual mode
        this.canvas = document.getElementById('imageCanvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        // Canvas - for review mode
        this.reviewCanvas = document.getElementById('reviewCanvas');
        this.reviewCtx = this.reviewCanvas ? this.reviewCanvas.getContext('2d') : null;

        // Canvas - for ROI overlay on review canvas
        this.roiOverlayCanvas = document.getElementById('roiOverlayCanvas');
        this.roiOverlayCtx = this.roiOverlayCanvas ? this.roiOverlayCanvas.getContext('2d') : null;

        this.searchInput = document.getElementById('searchInput');
        this.sortSelect = document.getElementById('sortSelect');
        
        this.processOcrAutoBtn = document.getElementById('processOcrAutoBtn');
        this.processOcrBtn = document.getElementById('processOcrBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.restartBtn = document.getElementById('restartBtn');

        // Lists & containers
        this.roiList = document.getElementById('roiList');
        this.resultsList = document.getElementById('resultsList');

        // Modal
        this.editModal = document.getElementById('editModal');
        this.editOriginal = document.getElementById('editOriginal');
        this.editCorrected = document.getElementById('editCorrected');
        this.editSaveBtn = document.getElementById('editSaveBtn');
        this.editCancelBtn = document.getElementById('editCancelBtn');

        // Steps
        this.stepUpload = document.getElementById('step-upload');
        this.stepRois = document.getElementById('step-rois');
        this.stepReview = document.getElementById('step-review');
    }

    attachEventListeners() {
        // Upload
        if (this.uploadArea) this.uploadArea.addEventListener('click', () => this.fileInput.click());
        if (this.fileInput) this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        if (this.uploadArea) this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        if (this.uploadArea) this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        if (this.uploadBtn) this.uploadBtn.addEventListener('click', () => this.uploadImage());

        // Mode selector
        if (this.autoDetectModeBtn) this.autoDetectModeBtn.addEventListener('click', () => this.switchMode('auto'));
        if (this.manualModeBtn) this.manualModeBtn.addEventListener('click', () => this.switchMode('manual'));

        // Canvas
        if (this.canvas) {
            this.canvas.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
            this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
            this.canvas.addEventListener('mouseup', (e) => this.handleCanvasMouseUp(e));
            this.canvas.addEventListener('touchstart', (e) => this.handleCanvasMouseDown(e));
            this.canvas.addEventListener('touchmove', (e) => this.handleCanvasMouseMove(e));
            this.canvas.addEventListener('touchend', (e) => this.handleCanvasMouseUp(e));
        }

        // ROI overlay canvas for clicking on grid cells
        if (this.roiOverlayCanvas) {
            this.roiOverlayCanvas.addEventListener('click', (e) => this.handleROIOverlayClick(e));
            this.roiOverlayCanvas.style.cursor = 'pointer';
        }

        // ROI buttons
        if (this.clearRoisBtn) this.clearRoisBtn.addEventListener('click', () => this.clearROIs());
        if (this.processOcrAutoBtn) this.processOcrAutoBtn.addEventListener('click', () => this.processOCRAuto());
        if (this.processOcrBtn) this.processOcrBtn.addEventListener('click', () => this.processOCR());

        // Results
        if (this.exportBtn) this.exportBtn.addEventListener('click', () => this.exportResults());
        if (this.restartBtn) this.restartBtn.addEventListener('click', () => this.restart());

        // Modal
        if (this.editSaveBtn) this.editSaveBtn.addEventListener('click', () => this.saveEdit());
        if (this.editCancelBtn) this.editCancelBtn.addEventListener('click', () => this.closeModal());
    }

    switchMode(mode) {
        if (mode === 'auto') {
            this.autoDetectMode.classList.remove('hidden');
            this.manualMode.classList.add('hidden');
            this.autoDetectModeBtn.classList.add('active');
            this.manualModeBtn.classList.remove('active');
        } else {
            this.autoDetectMode.classList.add('hidden');
            this.manualMode.classList.remove('hidden');
            this.autoDetectModeBtn.classList.remove('active');
            this.manualModeBtn.classList.add('active');
            this.renderCanvas();
        }
    }

    // ========== FILE UPLOAD ==========
    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.loadImage(files[0]);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadArea.classList.add('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            this.fileInput.files = files;
            this.loadImage(files[0]);
        }
    }

    loadImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.calculateScale();
                this.renderCanvas();
                this.uploadBtn.style.display = 'none';
                this.moveToStep('rois');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    calculateScale() {
        const maxWidth = 800;
        if (this.currentImage.width > maxWidth) {
            this.scale = maxWidth / this.currentImage.width;
            this.canvas.width = maxWidth;
            this.canvas.height = this.currentImage.height * this.scale;
        } else {
            this.scale = 1;
            this.canvas.width = this.currentImage.width;
            this.canvas.height = this.currentImage.height;
        }
    }

    async uploadImage() {
        if (!this.currentImage) {
            alert('Please select an image');
            return;
        }

        try {
            // Create session
            const sessionRes = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const session = await sessionRes.json();
            this.sessionId = session.session_id;

            // Upload image
            const canvas = document.createElement('canvas');
            canvas.width = this.currentImage.width;
            canvas.height = this.currentImage.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.currentImage, 0, 0);

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
            const formData = new FormData();
            formData.append('image', blob, 'image.jpg');

            const uploadRes = await fetch(`/api/sessions/${this.sessionId}/upload`, {
                method: 'POST',
                body: formData
            });

            if (uploadRes.ok) {
                this.moveToStep('rois');
            }
        } catch (error) {
            alert('Upload failed: ' + error.message);
        }
    }

    async loadImageDisplay() {
        // Load and display image from server on reviewCanvas
        console.log('loadImageDisplay called, sessionId:', this.sessionId);
        
        if (!this.sessionId) {
            console.warn('No sessionId, cannot load image');
            return;
        }
        
        if (!this.reviewCanvas || !this.reviewCtx) {
            console.warn('reviewCanvas or reviewCtx not found');
            return;
        }
        
        try {
            // First get session data to get original dimensions
            console.log('Fetching session data for dimensions');
            const sessionResponse = await fetch(`/api/sessions/${this.sessionId}`);
            if (sessionResponse.ok) {
                const sessionData = await sessionResponse.json();
                this.originalImageWidth = sessionData.metadata?.original_width || 0;
                this.originalImageHeight = sessionData.metadata?.original_height || 0;
                console.log('Original dimensions:', this.originalImageWidth, 'x', this.originalImageHeight);
            }
            
            console.log('Fetching image from /api/sessions/' + this.sessionId + '/image');
            const response = await fetch(`/api/sessions/${this.sessionId}/image`);
            console.log('Response status:', response.status);
            const data = await response.json();
            
            if (data.image) {
                console.log('Image data received, length:', data.image.length);
                const img = new Image();
                img.onload = () => {
                    console.log('Image loaded, dimensions:', img.width, 'x', img.height);
                    this.currentImage = img;
                    // Set canvas size to match image proportions
                    const maxWidth = 400;
                    const maxHeight = 500;
                    let width = img.width;
                    let height = img.height;
                    
                    const scale = Math.min(maxWidth / width, maxHeight / height);
                    width *= scale;
                    height *= scale;
                    
                    console.log('Canvas size set to:', width, 'x', height);
                    this.reviewCanvas.width = width;
                    this.reviewCanvas.height = height;
                    this.reviewCanvas.style.width = `${width}px`;
                    this.reviewCanvas.style.height = `${height}px`;

                    if (this.roiOverlayCanvas) {
                        this.roiOverlayCanvas.width = width;
                        this.roiOverlayCanvas.height = height;
                        this.roiOverlayCanvas.style.width = `${width}px`;
                        this.roiOverlayCanvas.style.height = `${height}px`;
                    }
                    
                    // Draw image
                    console.log('Drawing image on canvas');
                    this.reviewCtx.drawImage(img, 0, 0, width, height);
                    console.log('Image drawn successfully');
                    
                    // Draw ROI overlay after image is loaded
                    this.drawROIOverlay();
                };
                img.onerror = () => {
                    console.error('Failed to load image from data URL');
                };
                img.src = data.image;
            } else {
                console.warn('No image data in response');
            }
        } catch (error) {
            console.error('Failed to load image:', error);
        }
    }

    handleROIOverlayClick(e) {
        // Handle clicks on the ROI overlay to edit grid cells
        if (!this.roiOverlayCanvas || !this.originalImageWidth || !this.originalImageHeight) return;
        
        const rect = this.roiOverlayCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Convert canvas coordinates back to original image coordinates
        const scaleX = this.originalImageWidth / this.roiOverlayCanvas.width;
        const scaleY = this.originalImageHeight / this.roiOverlayCanvas.height;
        const originalX = x * scaleX;
        const originalY = y * scaleY;
        
        // Find which ROI was clicked
        for (let i = 0; i < this.results.length; i++) {
            const result = this.results[i];
            if (!result.roi || result.roi.length < 4) continue;
            
            const [x1, y1, x2, y2] = result.roi;
            if (originalX >= x1 && originalX <= x2 && originalY >= y1 && originalY <= y2) {
                // Found the clicked cell, open edit modal
                this.openModal(i, result.text || '');
                break;
            }
        }
    }

    // ========== CANVAS & ROI ==========
    renderCanvas() {
        if (!this.ctx || !this.currentImage) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(
            this.currentImage,
            0, 0,
            this.currentImage.width,
            this.currentImage.height,
            0, 0,
            this.canvas.width,
            this.canvas.height
        );

        // Draw ROIs
        this.drawROIs();

        // Draw current ROI being drawn
        if (this.currentROI) {
            this.drawROI(this.currentROI, '#ff9800', 3);
        }
    }

    drawROIs() {
        this.rois.forEach((roi, index) => {
            this.drawROI(roi, '#667eea', 2);
            
            // Draw number
            const x = roi[0] * this.scale + 5;
            const y = roi[1] * this.scale + 20;
            this.ctx.fillStyle = '#667eea';
            this.ctx.font = 'bold 14px Arial';
            this.ctx.fillText(String(index + 1), x, y);
        });
    }

    drawROI(roi, color, width) {
        const [x1, y1, x2, y2] = roi;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = width;
        this.ctx.fillStyle = color.replace(')', ', 0.1)').replace('rgb', 'rgba');
        this.ctx.fillRect(x1 * this.scale, y1 * this.scale, (x2 - x1) * this.scale, (y2 - y1) * this.scale);
        this.ctx.strokeRect(x1 * this.scale, y1 * this.scale, (x2 - x1) * this.scale, (y2 - y1) * this.scale);
    }

    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        let x, y;

        if (e.touches) {
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        } else {
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        }

        return {
            x: Math.round(x / this.scale),
            y: Math.round(y / this.scale)
        };
    }

    handleCanvasMouseDown(e) {
        if (!this.currentImage) return;

        const coords = this.getCanvasCoords(e);
        this.startX = coords.x;
        this.startY = coords.y;
        this.isDrawing = true;
        this.currentROI = null;
    }

    handleCanvasMouseMove(e) {
        if (!this.isDrawing || !this.currentImage) return;

        const coords = this.getCanvasCoords(e);
        const x1 = Math.min(this.startX, coords.x);
        const y1 = Math.min(this.startY, coords.y);
        const x2 = Math.max(this.startX, coords.x);
        const y2 = Math.max(this.startY, coords.y);

        this.currentROI = [x1, y1, x2, y2];
        this.renderCanvas();
    }

    handleCanvasMouseUp(e) {
        if (!this.isDrawing) return;

        this.isDrawing = false;

        if (this.currentROI) {
            const [x1, y1, x2, y2] = this.currentROI;
            const minSize = 10;
            if (Math.abs(x2 - x1) > minSize && Math.abs(y2 - y1) > minSize) {
                this.rois.push(this.currentROI);
                this.updateROIList();
            }
        }

        this.currentROI = null;
        this.renderCanvas();
    }

    updateROIList() {
        this.roiList.innerHTML = '';
        this.rois.forEach((roi, index) => {
            const item = document.createElement('div');
            item.className = 'roi-item';
            item.innerHTML = `
                <span class="roi-item-info">ROI ${index + 1}: ${roi[0]},${roi[1]} - ${roi[2]},${roi[3]}</span>
                <button class="roi-item-delete">Delete</button>
            `;
            item.querySelector('.roi-item-delete').addEventListener('click', () => {
                this.rois.splice(index, 1);
                this.updateROIList();
                this.renderCanvas();
            });
            this.roiList.appendChild(item);
        });
    }

    clearROIs() {
        this.rois = [];
        this.updateROIList();
        this.renderCanvas();
    }

    // ========== OCR PROCESSING ==========
    async processOCRAuto() {
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/ocr-auto`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) throw new Error('Auto-detection failed');

            const data = await response.json();
            this.results = data.results;
            this.displayResults();
            this.moveToStep('review');

        } catch (error) {
            alert('Auto-Detection Error: ' + error.message);
        }
    }

    async processOCR() {
        if (this.rois.length === 0) {
            alert('Please define at least one ROI');
            return;
        }

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/ocr`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rois: this.rois
                })
            });

            if (!response.ok) throw new Error('OCR processing failed');

            const data = await response.json();
            this.results = data.results;
            this.displayResults();
            this.moveToStep('review');

        } catch (error) {
            alert('OCR Error: ' + error.message);
        }
    }

    async importOCRResults(ocrBlocks) {
        // Import OCR results from contour_detection03.py
        // Args: ocrBlocks - List of {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float}
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/ocr-import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ocr_blocks: ocrBlocks
                })
            });

            if (!response.ok) throw new Error('Import failed');

            const data = await response.json();
            this.results = data.results;
            this.displayResults();
            this.moveToStep('review');

        } catch (error) {
            alert('Import Error: ' + error.message);
        }
    }

    displayResults() {
        // Load image first with a small delay to ensure DOM is ready
        setTimeout(() => {
            this.loadImageDisplay();
            this.drawROIOverlay();
        }, 100);
        
        this.renderResultCards();
        
        // Attach event listeners for filtering (after elements exist)
        const searchInput = document.getElementById('searchInput');
        const sortSelect = document.getElementById('sortSelect');
        
        if (searchInput) {
            searchInput.addEventListener('input', () => this.renderResultCards());
        }
        if (sortSelect) {
            sortSelect.addEventListener('change', () => this.renderResultCards());
        }
    }
    
    renderResultCards() {
        this.resultsList.innerHTML = '';
        
        // Get filtered and sorted results
        let displayResults = this.getFilteredResults();
        
        displayResults.forEach((result, displayIndex) => {
            const originalIndex = this.results.indexOf(result);
            const card = document.createElement('div');
            card.className = 'result-card-enhanced';
            
            if (result.is_corrected) {
                card.classList.add('corrected');
            }
            
            const originalText = result.text || '(empty)';
            const correctedText = result.corrected_text || originalText;
            
            card.innerHTML = `
                <div class="result-header-enhanced">
                    <span class="result-label">Cell ${originalIndex + 1}:</span>
                    <span style="flex: 1; font-weight: bold; color: #333;">${this.escapeHtml(originalText || '(empty)')}</span>
                    ${result.is_corrected ? '<span style="font-size: 0.8em; color: #10b981;">✓ Edited</span>' : ''}
                </div>
                
                <div class="result-content-enhanced">
                    <div class="result-original-text">Original OCR:</div>
                    <div class="result-original-value">${this.escapeHtml(originalText)}</div>
                    
                    <div style="font-size: 0.85em; color: #999; margin-top: 5px;">Corrected Value:</div>
                    <input type="text" class="result-input-enhanced" 
                           value="${this.escapeHtml(correctedText)}"
                           placeholder="Enter corrected value"
                           data-index="${originalIndex}">
                    
                    <div class="result-buttons-enhanced">
                        <button class="result-clear-btn" data-index="${originalIndex}">Clear</button>
                        <button class="result-reset-btn" data-index="${originalIndex}">Reset</button>
                    </div>
                </div>
            `;
            
            const input = card.querySelector('.result-input-enhanced');
            const clearBtn = card.querySelector('.result-clear-btn');
            const resetBtn = card.querySelector('.result-reset-btn');
            
            // Add click handler to the entire card for editing
            card.addEventListener('click', (e) => {
                // Don't trigger if clicking on buttons or input
                if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') {
                    this.openModal(originalIndex, result.text || '');
                }
            });
            
            // Hover highlighting on image
            card.addEventListener('mouseenter', () => {
                card.classList.add('highlighted');
                this.highlightROI(originalIndex, true);
            });
            
            card.addEventListener('mouseleave', () => {
                card.classList.remove('highlighted');
                this.highlightROI(originalIndex, false);
            });
            
            input.addEventListener('change', () => {
                this.updateResult(originalIndex, input.value);
                card.classList.add('corrected');
            });
            
            input.addEventListener('input', () => {
                card.classList.toggle('corrected', input.value !== originalText);
            });
            
            clearBtn.addEventListener('click', () => {
                input.value = '';
                this.updateResult(originalIndex, '');
                card.classList.remove('corrected');
            });
            
            resetBtn.addEventListener('click', () => {
                input.value = originalText;
                this.updateResult(originalIndex, originalText);
                card.classList.remove('corrected');
            });
            
            this.resultsList.appendChild(card);
        });
    }
    
    getFilteredResults() {
        let results = [...this.results];
        
        // Search filter
        if (this.searchInput && this.searchInput.value.trim()) {
            const searchTerm = this.searchInput.value.toLowerCase();
            results = results.filter(r => 
                r.text.toLowerCase().includes(searchTerm) || 
                (r.corrected_text && r.corrected_text.toLowerCase().includes(searchTerm))
            );
        }
        
        // Sorting
        if (this.sortSelect) {
            const sortType = this.sortSelect.value;
            if (sortType === 'edited') {
                results.sort((a, b) => (b.is_corrected ? 1 : 0) - (a.is_corrected ? 1 : 0));
            } else if (sortType === 'confidence') {
                results.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
            }
        }
        
        return results;
    }
    
    drawROIOverlay() {
        // Draw grid cell boxes on the overlay canvas with OCR text inside
        if (!this.roiOverlayCanvas || !this.roiOverlayCtx || !this.reviewCanvas) return;
        
        const canvas = this.roiOverlayCanvas;
        const ctx = this.roiOverlayCtx;
        
        // Match canvas size to reviewCanvas
        canvas.width = this.reviewCanvas.width;
        canvas.height = this.reviewCanvas.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scale from original image to displayed canvas
        if (!this.currentImage || !this.originalImageWidth || !this.originalImageHeight) return;
        
        const scaleX = canvas.width / this.originalImageWidth;
        const scaleY = canvas.height / this.originalImageHeight;
        
        const colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#52C4A0'
        ];
        
        // Draw grid cell boxes
        this.results.forEach((result, index) => {
            if (!result.roi || result.roi.length < 4) return;
            
            const [x1, y1, x2, y2] = result.roi;
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;
            
            const color = colors[index % colors.length];
            
            // Draw box outline
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            // Draw semi-transparent fill
            ctx.fillStyle = color + '15';  // Very light fill
            ctx.fillRect(x, y, w, h);
            
            // Draw OCR text inside the box
            const displayText = (result.corrected_text || result.text || '').trim();
            if (displayText) {
                ctx.fillStyle = '#000000';
                ctx.font = 'bold 12px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Fit text to box width
                let fontSize = 12;
                let textWidth = ctx.measureText(displayText).width;
                while (textWidth > w - 8 && fontSize > 8) {
                    fontSize--;
                    ctx.font = `bold ${fontSize}px Arial`;
                    textWidth = ctx.measureText(displayText).width;
                }
                
                ctx.fillText(displayText, x + w/2, y + h/2);
            } else {
                // Draw placeholder for empty cells
                ctx.fillStyle = '#666666';
                ctx.font = 'italic 10px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Click to edit', x + w/2, y + h/2);
            }
        });
    }
    
    highlightROI(index, isHighlight) {
        // Highlight a specific ROI on the canvas
        if (!this.roiOverlayCtx || !this.roiOverlayCanvas || !this.originalImageWidth || !this.originalImageHeight) return;
        
        const ctx = this.roiOverlayCtx;
        
        if (isHighlight) {
            // Redraw with highlight
            const result = this.results[index];
            if (!result || !result.roi) return;
            
            const [x1, y1, x2, y2] = result.roi;
            const scaleX = this.roiOverlayCanvas.width / this.originalImageWidth;
            const scaleY = this.roiOverlayCanvas.height / this.originalImageHeight;
            
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;
            
            // Draw pulsing glow effect
            ctx.shadowColor = '#FFD700';
            ctx.shadowBlur = 15;
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);
            ctx.shadowBlur = 0;
        } else {
            // Redraw entire overlay
            this.drawROIOverlay();
        }
    }

    updateResult(index, correctedText) {
        this.results[index].corrected_text = correctedText;
        this.results[index].is_corrected = correctedText !== this.results[index].text;
    }

    // ========== MODAL EDITING ==========
    openModal(index, originalText) {
        this.editingIndex = index;
        const result = this.results[index];
        const currentText = result.corrected_text || result.text || '';
        
        this.editOriginal.value = originalText || '(empty)';
        this.editCorrected.value = currentText;
        this.editModal.classList.remove('hidden');
        
        // Focus on the input field
        this.editCorrected.focus();
        this.editCorrected.select();
    }

    closeModal() {
        this.editModal.classList.add('hidden');
        this.editingIndex = null;
    }

    saveEdit() {
        if (this.editingIndex !== null) {
            this.updateResult(this.editingIndex, this.editCorrected.value);
            this.displayResults();
        }
        this.closeModal();
    }

    // ========== EXPORT ==========
    async exportResults() {
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/export`);
            const data = await response.json();

            // Create JSON file
            const json = JSON.stringify(data, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ocr_results_${this.sessionId}.json`;
            a.click();

            alert('Results exported successfully!');
        } catch (error) {
            alert('Export failed: ' + error.message);
        }
    }

    // ========== UTILITIES ==========
    moveToStep(step) {
        this.stepUpload.classList.add('hidden');
        this.stepRois.classList.add('hidden');
        this.stepReview.classList.add('hidden');

        if (step === 'upload') this.stepUpload.classList.remove('hidden');
        if (step === 'rois') this.stepRois.classList.remove('hidden');
        if (step === 'review') this.stepReview.classList.remove('hidden');
    }

    restart() {
        this.sessionId = null;
        this.currentImage = null;
        this.rois = [];
        this.results = [];
        this.fileInput.value = '';
        this.roiList.innerHTML = '';
        this.resultsList.innerHTML = '';
        this.moveToStep('upload');
        this.uploadBtn.style.display = 'inline-block';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.editor = new OCREditor();
});
