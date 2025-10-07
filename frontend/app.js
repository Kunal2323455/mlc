// ===== TULSI AI - Advanced Plant Disease Detection Frontend =====

// DOM Elements
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const browseBtn = document.getElementById('browseBtn');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const preview = document.getElementById('preview');
const resultSection = document.getElementById('resultSection');
const summary = document.getElementById('summary');
const probs = document.getElementById('probs');
const recommendation = document.getElementById('recommendation');
const loading = document.getElementById('loading');
const toasts = document.getElementById('toasts');
const themeToggle = document.getElementById('themeToggle');
const confidencePill = document.getElementById('confidencePill');
const modelSelect = document.getElementById('modelSelect');
const modelInfo = document.getElementById('modelInfo');
const modelGrid = document.getElementById('modelGrid');
const modelStatus = document.getElementById('modelStatus');
const modelUsedBadge = document.getElementById('modelUsedBadge');
const infoBtn = document.getElementById('infoBtn');
const infoModal = document.getElementById('infoModal');
const closeModal = document.getElementById('closeModal');
const copyBtn = document.getElementById('copyBtn');

// Additional result elements
const diagnosisDetails = document.getElementById('diagnosisDetails');
const primaryDiagnosis = document.getElementById('primaryDiagnosis');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceAnalysis = document.getElementById('confidenceAnalysis');
const modelPerformance = document.getElementById('modelPerformance');

// Global variables
let selectedFile = null;
let availableModels = [];
let modelDetails = {};

// Model information database
const MODEL_INFO = {
    'custom_cnn': {
        name: 'Custom CNN',
        description: 'Deep convolutional network designed specifically for plant diseases',
        accuracy: '87-92%',
        type: 'Custom Architecture',
        speed: 'Fast',
        details: 'Multi-layer CNN with batch normalization and dropout regularization'
    },
    'vgg16_transfer': {
        name: 'VGG16 Transfer',
        description: 'Classic deep learning architecture with transfer learning',
        accuracy: '89-94%',
        type: 'Transfer Learning',
        speed: 'Medium',
        details: 'Pre-trained VGG16 with custom classification layers'
    },
    'mobilenet_transfer': {
        name: 'MobileNet Transfer',
        description: 'Lightweight model optimized for mobile deployment',
        accuracy: '88-93%',
        type: 'Mobile Optimized',
        speed: 'Very Fast',
        details: 'Efficient architecture with depthwise separable convolutions'
    },
    'efficientnet_transfer': {
        name: 'EfficientNet Transfer',
        description: 'State-of-the-art efficient neural network architecture',
        accuracy: '92-97%',
        type: 'State-of-the-art',
        speed: 'Medium',
        details: 'Compound scaling method for optimal efficiency and accuracy'
    },
    'resnet50_transfer': {
        name: 'ResNet50 Transfer',
        description: 'Deep residual network with skip connections',
        accuracy: '90-95%',
        type: 'Deep Learning',
        speed: 'Medium',
        details: 'Residual connections enable training of very deep networks'
    },
    'ensemble_weighted': {
        name: 'Ensemble Model',
        description: 'Combines multiple models for maximum accuracy',
        accuracy: '95-98%',
        type: 'Ensemble',
        speed: 'Slow',
        details: 'Weighted combination of top-performing models'
    }
};

// Initialize application
window.addEventListener('load', () => {
    showLoading(false);
    loadModels();
    initializeTheme();
    setupEventListeners();
});

// ===== MODEL MANAGEMENT =====

async function loadModels() {
    try {
        modelStatus.textContent = 'Loading...';
        modelStatus.className = 'status-badge loading';
        
        const res = await fetch((window.API_BASE || '') + '/models');
        if (!res.ok) throw new Error('Could not load models.');
        
        const data = await res.json();
        availableModels = data.models || [];
        
        populateModelSelect();
        createModelGrid();
        
        modelStatus.textContent = `${availableModels.length} Models Ready`;
        modelStatus.className = 'status-badge ready';
        
        toast(`Loaded ${availableModels.length} AI models successfully`, false);
        
    } catch (err) {
        console.error('Error loading models:', err);
        modelStatus.textContent = 'Error Loading';
        modelStatus.className = 'status-badge error';
        
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        toast(err.message || 'Failed to load models', true);
    }
}

function populateModelSelect() {
    modelSelect.innerHTML = '<option value="">Select an AI model...</option>';
    
    availableModels.forEach(modelKey => {
        const option = document.createElement('option');
        option.value = modelKey;
        
        const info = MODEL_INFO[modelKey] || {};
        option.textContent = info.name || formatModelName(modelKey);
        
        modelSelect.appendChild(option);
    });
}

function createModelGrid() {
    modelGrid.innerHTML = '';
    
    availableModels.forEach(modelKey => {
        const info = MODEL_INFO[modelKey] || {};
        const card = createModelCard(modelKey, info);
        modelGrid.appendChild(card);
    });
}

function createModelCard(modelKey, info) {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.model = modelKey;
    
    card.innerHTML = `
        <h4>${info.name || formatModelName(modelKey)}</h4>
        <p>${info.description || 'Advanced AI model for plant disease detection'}</p>
        <div class="model-stats">
            <span class="model-accuracy">${info.accuracy || '85-90%'}</span>
            <span class="model-type">${info.type || 'Neural Network'}</span>
        </div>
    `;
    
    card.addEventListener('click', () => selectModel(modelKey, card));
    
    return card;
}

function selectModel(modelKey, cardElement) {
    // Update select dropdown
    modelSelect.value = modelKey;
    
    // Update visual selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    cardElement.classList.add('selected');
    
    // Show model info
    showModelInfo(modelKey);
    
    // Enable prediction if image is loaded
    updatePredictButton();
}

function showModelInfo(modelKey) {
    const info = MODEL_INFO[modelKey] || {};
    
    if (modelInfo && info.name) {
        const accuracyBadge = modelInfo.querySelector('#modelAccuracy') || modelInfo.querySelector('.accuracy-badge');
        const descElement = modelInfo.querySelector('#modelDescription') || modelInfo.querySelector('.model-desc');
        
        if (accuracyBadge) {
            accuracyBadge.textContent = `Accuracy: ${info.accuracy}`;
            accuracyBadge.className = 'accuracy-badge';
        }
        
        if (descElement) {
            descElement.textContent = info.details || info.description;
        }
        
        modelInfo.classList.remove('hidden');
    }
}

function formatModelName(modelKey) {
    return modelKey
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
        .replace('Cnn', 'CNN');
}

// ===== EVENT LISTENERS =====

function setupEventListeners() {
    // File input events
    dropzone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });
    
    // Drag and drop events
    dropzone.addEventListener('dragover', handleDragOver);
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);
    
    // File selection
    fileInput.addEventListener('change', handleFileSelect);
    
    // Model selection
    modelSelect.addEventListener('change', handleModelSelectChange);
    
    // Control buttons
    predictBtn.addEventListener('click', predict);
    clearBtn.addEventListener('click', clearSelection);
    copyBtn.addEventListener('click', copyRecommendation);
    
    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);
    
    // Info modal
    infoBtn.addEventListener('click', () => infoModal.classList.remove('hidden'));
    closeModal.addEventListener('click', () => infoModal.classList.add('hidden'));
    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) infoModal.classList.add('hidden');
    });
    
    // Recommendation click to copy
    recommendation.addEventListener('click', copyRecommendation);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

// ===== FILE HANDLING =====

function handleDragOver(e) {
    e.preventDefault();
    dropzone.classList.add('dragover');
}

function handleDragLeave() {
    dropzone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        setPreview(file);
        resetResults();
        updatePredictButton();
    } else if (file) {
        toast('Please select a valid image file (JPG, PNG, JPEG)', true);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        setPreview(file);
        resetResults();
        updatePredictButton();
    } else if (file) {
        toast('Please select a valid image file', true);
    }
}

function handleModelSelectChange() {
    const selectedModel = modelSelect.value;
    
    // Update model card selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
        if (card.dataset.model === selectedModel) {
            card.classList.add('selected');
        }
    });
    
    if (selectedModel) {
        showModelInfo(selectedModel);
    } else {
        modelInfo.classList.add('hidden');
    }
    
    updatePredictButton();
}

function setPreview(file) {
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove('hidden');
    dropzone.querySelector('.placeholder').classList.add('hidden');
    clearBtn.disabled = false;
    
    preview.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    preview.classList.add('hidden');
    dropzone.querySelector('.placeholder').classList.remove('hidden');
    clearBtn.disabled = true;
    resetResults();
    updatePredictButton();
}

function updatePredictButton() {
    const hasImage = selectedFile !== null;
    const hasModel = modelSelect.value !== '';
    
    predictBtn.disabled = !(hasImage && hasModel);
    
    if (!hasImage) {
        predictBtn.innerHTML = '<span class="btn-icon">📷</span>Upload Image First';
    } else if (!hasModel) {
        predictBtn.innerHTML = '<span class="btn-icon">🧠</span>Select AI Model';
    } else {
        predictBtn.innerHTML = '<span class="btn-icon">🔍</span>Analyze with AI';
    }
}

// ===== PREDICTION =====

async function predict() {
    if (!selectedFile) {
        toast('Please select an image first', true);
        return;
    }
    
    const selectedModel = modelSelect.value;
    if (!selectedModel) {
        toast('Please select an AI model first', true);
        return;
    }
    
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    showLoading(true);
    
    try {
        const form = new FormData();
        form.append('file', selectedFile);
        form.append('model_name', selectedModel);
        
        const res = await fetch((window.API_BASE || '') + '/predict', {
            method: 'POST',
            body: form
        });
        
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Server error' }));
            throw new Error(err.detail || 'Prediction failed');
        }
        
        const data = await res.json();
        renderResult(data);
        toast('Analysis complete! 🎉', false);
        
    } catch (err) {
        console.error('Prediction error:', err);
        toast(err.message || 'Prediction failed. Please try again.', true);
    } finally {
        showLoading(false);
        predictBtn.disabled = false;
        clearBtn.disabled = false;
    }
}

// ===== RESULTS RENDERING =====

function renderResult(data) {
    resultSection.classList.remove('hidden');
    
    const pred = data.prediction || {};
    const disease = pred.predicted_disease || 'Unknown';
    const conf = Number(pred.confidence || 0);
    const confPct = (conf * 100).toFixed(1) + '%';
    const isHealthy = !!pred.is_healthy;
    const modelUsed = data.model_used || 'Unknown';
    
    // Update summary
    const modelInfo = MODEL_INFO[modelUsed] || {};
    summary.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <strong>Diagnosis:</strong> ${disease}
                <div style="font-size: 0.875rem; color: var(--text-muted); margin-top: 0.25rem;">
                    Analyzed with ${modelInfo.name || formatModelName(modelUsed)}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.875rem; color: var(--text-muted);">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">${confPct}</div>
            </div>
        </div>
    `;
    
    // Update badges
    confidencePill.textContent = `Confidence: ${confPct}`;
    confidencePill.className = `pill ${getConfidenceClass(conf)}`;
    confidencePill.classList.remove('hidden');
    
    modelUsedBadge.textContent = modelInfo.name || formatModelName(modelUsed);
    modelUsedBadge.classList.remove('hidden');
    
    // Update diagnosis details
    updateDiagnosisDetails(disease, conf, isHealthy);
    
    // Update probabilities
    updateProbabilities(pred.all_class_probabilities || {}, disease);
    
    // Update recommendation
    recommendation.textContent = data.recommendation || 'No recommendation available.';
    
    // Update analysis extras
    updateAnalysisExtras(pred, modelUsed, conf);
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateDiagnosisDetails(disease, confidence, isHealthy) {
    const diseaseClass = disease.toLowerCase();
    const colors = {
        'bacterial': '#ef4444',
        'fungal': '#f59e0b', 
        'pests': '#8b5cf6',
        'healthy': '#10b981'
    };
    
    primaryDiagnosis.innerHTML = `
        <div style="color: ${colors[diseaseClass] || '#64748b'}; font-size: 1.25rem;">
            ${getDiseaseIcon(diseaseClass)} ${disease}
        </div>
    `;
    
    // Update confidence bar
    const fill = confidenceBar.querySelector('.confidence-fill');
    const text = confidenceBar.querySelector('.confidence-text');
    
    if (fill && text) {
        setTimeout(() => {
            fill.style.width = `${confidence * 100}%`;
        }, 100);
        
        text.textContent = `${(confidence * 100).toFixed(1)}% Confident`;
    }
}

function updateProbabilities(probabilities, predictedClass) {
    probs.innerHTML = '';
    
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([cls, prob], index) => {
        const pct = Math.round(prob * 100);
        const isTop = cls === predictedClass;
        
        const row = document.createElement('div');
        row.className = `prob-item ${isTop ? 'top-prediction' : ''}`;
        
        row.innerHTML = `
            <span style="display: flex; align-items: center; gap: 0.5rem;">
                ${getDiseaseIcon(cls.toLowerCase())}
                <span style="text-transform: capitalize; font-weight: ${isTop ? '700' : '500'};">${cls}</span>
            </span>
            <div class="bar">
                <span style="background: ${getBarColor(cls.toLowerCase())};"></span>
            </div>
            <span style="font-weight: 600; color: ${isTop ? 'var(--primary-color)' : 'var(--text-secondary)'};">${pct}%</span>
        `;
        
        probs.appendChild(row);
        
        // Animate bar
        requestAnimationFrame(() => {
            const bar = row.querySelector('.bar > span');
            if (bar) {
                setTimeout(() => {
                    bar.style.width = pct + '%';
                }, index * 100);
            }
        });
    });
}

function updateAnalysisExtras(prediction, modelUsed, confidence) {
    // Confidence analysis
    const confidenceLevel = getConfidenceLevel(confidence);
    confidenceAnalysis.innerHTML = `
        <div style="margin-bottom: 0.75rem;">
            <strong>Confidence Level:</strong> ${confidenceLevel.label}
        </div>
        <div style="font-size: 0.875rem; line-height: 1.5;">
            ${confidenceLevel.description}
        </div>
    `;
    
    // Model performance
    const modelInfo = MODEL_INFO[modelUsed] || {};
    modelPerformance.innerHTML = `
        <div style="margin-bottom: 0.75rem;">
            <strong>Model:</strong> ${modelInfo.name || formatModelName(modelUsed)}
        </div>
        <div style="font-size: 0.875rem; line-height: 1.5;">
            <div><strong>Type:</strong> ${modelInfo.type || 'Neural Network'}</div>
            <div><strong>Accuracy:</strong> ${modelInfo.accuracy || '85-90%'}</div>
            <div><strong>Speed:</strong> ${modelInfo.speed || 'Medium'}</div>
        </div>
    `;
}

// ===== UTILITY FUNCTIONS =====

function getDiseaseIcon(disease) {
    const icons = {
        'bacterial': '🦠',
        'fungal': '🍄',
        'pests': '🐛',
        'healthy': '✅'
    };
    return icons[disease] || '❓';
}

function getBarColor(disease) {
    const colors = {
        'bacterial': '#ef4444',
        'fungal': '#f59e0b',
        'pests': '#8b5cf6', 
        'healthy': '#10b981'
    };
    return colors[disease] || '#64748b';
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high-confidence';
    if (confidence >= 0.6) return 'medium-confidence';
    return 'low-confidence';
}

function getConfidenceLevel(confidence) {
    if (confidence >= 0.9) {
        return {
            label: 'Very High',
            description: 'The model is very confident in this prediction. The diagnosis is highly reliable.'
        };
    } else if (confidence >= 0.8) {
        return {
            label: 'High', 
            description: 'The model shows high confidence. This is a reliable diagnosis.'
        };
    } else if (confidence >= 0.6) {
        return {
            label: 'Medium',
            description: 'The model has moderate confidence. Consider getting a second opinion or additional testing.'
        };
    } else {
        return {
            label: 'Low',
            description: 'The model has low confidence in this prediction. Additional analysis or expert consultation is recommended.'
        };
    }
}

function resetResults() {
    resultSection.classList.add('hidden');
    summary.innerHTML = '';
    probs.innerHTML = '';
    recommendation.textContent = '';
    confidencePill.classList.add('hidden');
    modelUsedBadge.classList.add('hidden');
}

// ===== UI FUNCTIONS =====

function showLoading(show) {
    loading.style.display = show ? 'grid' : 'none';
    loading.setAttribute('aria-hidden', String(!show));
}

function toast(message, isError = false) {
    const el = document.createElement('div');
    el.className = `toast ${isError ? 'error' : ''}`;
    el.textContent = message;
    
    toasts.appendChild(el);
    
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, isError ? 5000 : 3000);
}

async function copyRecommendation() {
    const text = recommendation.textContent || '';
    if (!text.trim()) {
        toast('No recommendation to copy', true);
        return;
    }
    
    try {
        await navigator.clipboard.writeText(text);
        toast('Recommendation copied to clipboard! 📋', false);
        
        // Visual feedback
        copyBtn.textContent = '✓ Copied';
        setTimeout(() => {
            copyBtn.textContent = '📋 Copy';
        }, 2000);
        
    } catch (err) {
        console.error('Copy failed:', err);
        toast('Failed to copy to clipboard', true);
    }
}

// ===== THEME MANAGEMENT =====

function initializeTheme() {
    const savedTheme = localStorage.getItem('tulsi-theme');
    if (savedTheme === 'light') {
        document.documentElement.classList.add('light');
        themeToggle.textContent = '☀️';
    } else {
        themeToggle.textContent = '🌙';
    }
}

function toggleTheme() {
    const isLight = document.documentElement.classList.toggle('light');
    themeToggle.textContent = isLight ? '☀️' : '🌙';
    localStorage.setItem('tulsi-theme', isLight ? 'light' : 'dark');
    
    toast(`Switched to ${isLight ? 'light' : 'dark'} theme`, false);
}

// ===== KEYBOARD SHORTCUTS =====

function handleKeyboard(e) {
    // Ctrl/Cmd + U: Upload image
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        fileInput.click();
    }
    
    // Ctrl/Cmd + Enter: Predict
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!predictBtn.disabled) {
            predict();
        }
    }
    
    // Ctrl/Cmd + R: Clear (prevent default refresh)
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        clearSelection();
    }
    
    // Escape: Close modal
    if (e.key === 'Escape') {
        infoModal.classList.add('hidden');
    }
    
    // Ctrl/Cmd + C: Copy recommendation (when results are visible)
    if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !resultSection.classList.contains('hidden')) {
        if (recommendation.textContent.trim()) {
            e.preventDefault();
            copyRecommendation();
        }
    }
}

// ===== ERROR HANDLING =====

window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    toast('An unexpected error occurred. Please refresh the page.', true);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    toast('A network error occurred. Please check your connection.', true);
});

// ===== PERFORMANCE MONITORING =====

if ('performance' in window) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            if (perfData) {
                console.log(`Page loaded in ${Math.round(perfData.loadEventEnd - perfData.fetchStart)}ms`);
            }
        }, 0);
    });
}

// ===== SERVICE WORKER (for offline functionality) =====

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(() => console.log('Service Worker registered'))
            .catch(() => console.log('Service Worker registration failed'));
    });
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        loadModels,
        predict,
        formatModelName,
        getDiseaseIcon,
        getConfidenceLevel
    };
}