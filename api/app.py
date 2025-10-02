"""
Flask API for Text Classification
This module provides a REST API for text classification predictions.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    from prediction import TextClassificationPredictor
    print("✓ Successfully imported TextClassificationPredictor")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Source path exists: {os.path.exists(src_path)}")
    if os.path.exists(src_path):
        print(f"Files in src: {os.listdir(src_path)}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor with error handling."""
    global predictor
    try:
        # Get the directory of this file and build paths relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'models', 'best_model.pkl')
        preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
        
        predictor = TextClassificationPredictor(model_path, preprocessor_path)
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classification API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .prediction-details {
            margin-top: 10px;
        }
        .confidence-bar {
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin-top: 5px;
            overflow: hidden;
        }
        .confidence-fill {
            background-color: #28a745;
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .examples {
            margin-top: 30px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .examples h3 {
            margin-top: 0;
            color: #495057;
        }
        .example-text {
            background-color: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid #dee2e6;
        }
        .example-text:hover {
            background-color: #e9ecef;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 10px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Text Classification API</h1>
        
        <div class="form-group">
            <label for="text-input">Enter text to classify:</label>
            <textarea id="text-input" placeholder="Type or paste your text here..."></textarea>
        </div>
        
        <button onclick="classifyText()">Classify Text</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Classifying...</p>
        </div>
        
        <div class="result" id="result">
            <div id="result-content"></div>
        </div>
        
        <div class="examples">
            <h3>Example Texts (Click to Try):</h3>
            <div class="example-text" onclick="setExample(this)">I love playing basketball and watching NBA games every weekend</div>
            <div class="example-text" onclick="setExample(this)">The new graphics processing unit is incredible for gaming and machine learning</div>
            <div class="example-text" onclick="setExample(this)">Prayer and faith have always been important aspects of my spiritual life</div>
            <div class="example-text" onclick="setExample(this)">The patient needs immediate medical attention for acute chest pain symptoms</div>
            <div class="example-text" onclick="setExample(this)">Atheism is simply the lack of belief in deities or supernatural beings</div>
        </div>
    </div>

    <script>
        function setExample(element) {
            document.getElementById('text-input').value = element.textContent;
        }
        
        async function classifyText() {
            const text = document.getElementById('text-input').value.trim();
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('result-content');
            const loading = document.getElementById('loading');
            
            if (!text) {
                showResult('Please enter some text to classify.', false);
                return;
            }
            
            // Show loading
            loading.style.display = 'block';
            resultDiv.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text, return_probabilities: true })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayPrediction(data);
                } else {
                    showResult(`Error: ${data.error}`, false);
                }
            } catch (error) {
                showResult(`Error: ${error.message}`, false);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function displayPrediction(data) {
            let html = `
                <h3>📊 Prediction Results</h3>
                <div class="prediction-details">
                    <p><strong>Predicted Category:</strong> ${data.predicted_category}</p>
                    <p><strong>Model Used:</strong> ${data.model_used}</p>
                    <p><strong>Model Accuracy:</strong> ${(data.model_accuracy * 100).toFixed(2)}%</p>
            `;
            
            if (data.confidence) {
                html += `
                    <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                `;
            }
            
            if (data.probabilities) {
                html += '<p><strong>Top 3 Predictions:</strong></p><ul>';
                const topPredictions = Object.entries(data.probabilities).slice(0, 3);
                topPredictions.forEach(([category, prob]) => {
                    html += `<li>${category}: ${(prob * 100).toFixed(2)}%</li>`;
                });
                html += '</ul>';
            }
            
            html += '</div>';
            
            document.getElementById('result-content').innerHTML = html;
            showResult('', true);
        }
        
        function showResult(message, isSuccess) {
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('result-content');
            
            if (message) {
                resultContent.innerHTML = message;
            }
            
            resultDiv.className = `result ${isSuccess ? 'success' : 'error'}`;
            resultDiv.style.display = 'block';
        }
        
        // Allow Enter key to submit
        document.getElementById('text-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                classifyText();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global predictor
    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'Predictor not initialized'
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Text classification API is running',
        'model': predictor.model_name if predictor else None,
        'accuracy': predictor.model_accuracy if predictor else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction on input text."""
    global predictor
    
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized. Please check if model files exist.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text'].strip()
        return_probabilities = data.get('return_probabilities', False)
        
        if not text:
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Make prediction
        result = predictor.predict_single(text, return_probabilities)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Make predictions on multiple texts."""
    global predictor
    
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized. Please check if model files exist.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts (should be a list)'
            }), 400
        
        texts = data['texts']
        return_probabilities = data.get('return_probabilities', False)
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts should be a list of strings'
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                'error': 'texts list cannot be empty'
            }), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({
                'error': 'Maximum batch size is 100 texts'
            }), 400
        
        # Make predictions
        results = predictor.predict_batch(texts, return_probabilities)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories."""
    global predictor
    
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized'
        }), 500
    
    return jsonify({
        'categories': predictor.target_names,
        'count': len(predictor.target_names)
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information."""
    global predictor
    
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized'
        }), 500
    
    return jsonify({
        'model_name': predictor.model_name,
        'model_accuracy': predictor.model_accuracy,
        'categories': predictor.target_names,
        'num_categories': len(predictor.target_names)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'POST /predict',
            'POST /predict/batch',
            'GET /categories',
            'GET /model/info'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check server logs for details'
    }), 500

def main():
    """Main function to run the Flask app."""
    print("Initializing Text Classification API...")
    
    if not initialize_predictor():
        print("Failed to initialize predictor. Please ensure model files exist.")
        print("Run 'python src/model_training.py' to train models first.")
        return
    
    print(f"Predictor initialized successfully!")
    print(f"Model: {predictor.model_name}")
    print(f"Accuracy: {predictor.model_accuracy:.4f}")
    print(f"Categories: {len(predictor.target_names)}")
    
    print("\nStarting Flask API server...")
    print("Available endpoints:")
    print("  GET  /           - Web interface")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("  GET  /categories - Available categories")
    print("  GET  /model/info - Model information")
    
    print(f"\nAPI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()