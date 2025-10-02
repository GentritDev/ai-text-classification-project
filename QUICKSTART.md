# Quick Start Guide

## Installation and Setup

### 1. Install Dependencies
```powershell
# Navigate to project directory
cd "c:\Roadmap 2025\Intern\ai - text classification task"

# Install required packages
pip install -r requirements.txt

# Or run the setup script
python setup.py
```

### 2. Run the Complete Pipeline
```powershell
# Run the main script to execute the full pipeline
python main.py
```

This will:
- Load and preprocess the 20 Newsgroups dataset
- Train and compare multiple ML models
- Create comprehensive visualizations
- Save trained models for later use
- Demonstrate predictions on sample texts

### 3. Start the Flask API
```powershell
# Start the REST API server
python api/app.py
```

Then visit `http://localhost:5000` for the web interface.

### 4. Use Individual Components

**Data Preprocessing:**
```powershell
python src/data_preprocessing.py
```

**Model Training:**
```powershell
python src/model_training.py
```

**Make Predictions:**
```powershell
python src/prediction.py
```

**Create Visualizations:**
```powershell
python src/visualization.py
```

### 5. Jupyter Notebook
```powershell
# Start Jupyter and open the exploration notebook
jupyter notebook notebooks/text_classification_exploration.ipynb
```

## API Usage Examples

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love playing basketball and watching NBA games", "return_probabilities": true}'
```

### Batch Predictions
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sports are fun", "Computer graphics are amazing"], "return_probabilities": true}'
```

### Get Available Categories
```bash
curl http://localhost:5000/categories
```

## Project Features

### ✅ Core Requirements
- [x] Data loading and preprocessing (20 Newsgroups dataset)
- [x] Text cleaning, stopword removal, tokenization
- [x] Exploratory data analysis with statistics
- [x] Multiple model training (Logistic Regression, Naive Bayes, SVM, Random Forest)
- [x] Train/test split and model evaluation
- [x] Accuracy, precision, recall, F1-score metrics
- [x] Prediction script for new text classification

### ✅ Bonus Features
- [x] TF-IDF vectorization (advanced text representation)
- [x] Model comparison with detailed analysis
- [x] Comprehensive visualizations (confusion matrix, word clouds, charts)
- [x] Flask REST API with web interface
- [x] Cross-validation and hyperparameter tuning
- [x] Interactive Jupyter notebook
- [x] Batch prediction support
- [x] Comprehensive documentation

## File Structure
```
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_training.py        # Model training and evaluation
│   ├── prediction.py            # Prediction interface
│   └── visualization.py         # Visualization suite
├── api/
│   └── app.py                   # Flask REST API
├── notebooks/
│   └── text_classification_exploration.ipynb  # Interactive notebook
├── models/                      # Trained models (created after training)
├── visualizations/              # Generated plots (created after visualization)
├── main.py                      # Main execution script
├── setup.py                     # Setup and installation script
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

## Troubleshooting

**If models are not found:**
1. Run `python main.py` or `python src/model_training.py` first
2. Ensure the `models/` directory contains the generated `.pkl` files

**If NLTK data is missing:**
1. Run `python setup.py` to download required NLTK data
2. Or manually download: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

**If API fails to start:**
1. Ensure models are trained and saved
2. Check that all dependencies are installed
3. Verify the Flask app can find the model files

## Next Steps
1. Push to GitHub repository
2. Test all functionality
3. Send repository link to office@linkplus-it.com
4. Consider deploying to cloud platform for production use