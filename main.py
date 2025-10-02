"""
Main Script for Text Classification Project
This script runs the complete pipeline: data loading, preprocessing, training, and evaluation.
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import TextPreprocessor
from src.model_training import ModelTrainer
from src.prediction import TextClassificationPredictor
from src.visualization import TextClassificationVisualizer

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*50}")
    print(f"{title}")
    print(f"{'-'*50}")

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'visualizations', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_data_preprocessing():
    """Run data preprocessing pipeline."""
    print_section("Data Preprocessing")
    
    preprocessor = TextPreprocessor()
    
    # Use subset of categories for demonstration (remove this line to use all 20 categories)
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    
    print(f"Loading dataset with categories: {categories}")
    dataset = preprocessor.load_data(categories=categories)
    
    print(f"Dataset loaded: {len(dataset['data'])} samples, {len(dataset['target_names'])} categories")
    
    # Get and display statistics
    stats = preprocessor.get_dataset_statistics(
        dataset['data'], dataset['target'], dataset['target_names']
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Categories: {stats['num_categories']}")
    print(f"  Average text length: {stats['avg_text_length']:.1f} words")
    
    print(f"\nCategory distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    # Preprocess subset for demonstration
    sample_size = min(2000, len(dataset['data']))  # Use sample for faster processing
    print(f"\nPreprocessing {sample_size} samples...")
    
    preprocessed_texts = preprocessor.preprocess_texts(
        dataset['data'][:sample_size], 
        remove_stopwords=True, 
        apply_stemming=False
    )
    
    # Create TF-IDF features
    features = preprocessor.create_tfidf_features(preprocessed_texts)
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print(f"✓ Preprocessing completed successfully!")
    print(f"✓ Feature matrix shape: {features.shape}")
    print(f"✓ Preprocessor saved to models/preprocessor.pkl")
    
    return True

def run_model_training():
    """Run model training pipeline."""
    print_section("Model Training")
    
    trainer = ModelTrainer()
    
    # Use subset of categories for demonstration
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    
    print(f"Training models on categories: {categories}")
    print("Note: Using subset for demonstration. Remove 'categories' parameter to use all 20 categories.")
    
    # Train all models with hyperparameter tuning
    results = trainer.train_all_models(categories=categories, perform_tuning=True)
    
    # Save models
    trainer.save_models()
    
    print(f"✓ Model training completed successfully!")
    print(f"✓ Best model: {trainer.best_model_name}")
    print(f"✓ Best accuracy: {trainer.model_scores[trainer.best_model_name]:.4f}")
    print(f"✓ All models saved in models/ directory")
    
    return results

def run_visualization():
    """Run visualization pipeline."""
    print_section("Creating Visualizations")
    
    visualizer = TextClassificationVisualizer()
    
    # Use same categories as training
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    
    print("Creating comprehensive visualization report...")
    visualizer.create_comprehensive_report(categories=categories)
    
    print(f"✓ Visualizations created successfully!")
    print(f"✓ All plots saved in visualizations/ directory")
    
    return True

def run_prediction_demo():
    """Run prediction demonstration."""
    print_section("Prediction Demonstration")
    
    try:
        predictor = TextClassificationPredictor()
        
        print(f"Loaded model: {predictor.model_name}")
        print(f"Model accuracy: {predictor.model_accuracy:.4f}")
        print(f"Available categories: {len(predictor.target_names)}")
        
        # Sample texts for demonstration
        sample_texts = [
            "I love playing hockey and watching NHL games every weekend",
            "The new graphics processing unit has amazing performance for gaming",
            "Prayer and meditation help me find inner peace and spiritual connection",
            "The patient shows symptoms of acute respiratory infection",
            "I don't believe in any supernatural deities or religious concepts"
        ]
        
        print(f"\nDemonstrating predictions on {len(sample_texts)} sample texts:")
        print("-" * 70)
        
        for i, text in enumerate(sample_texts, 1):
            result = predictor.predict_single(text, return_probabilities=True)
            
            print(f"\nSample {i}: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"Prediction: {result['predicted_category']}")
            if result.get('confidence'):
                print(f"Confidence: {result['confidence']:.4f}")
            
            # Show top 2 predictions
            if result.get('probabilities'):
                top_2 = list(result['probabilities'].items())[:2]
                print(f"Top predictions: {top_2[0][0]} ({top_2[0][1]:.3f}), {top_2[1][0]} ({top_2[1][1]:.3f})")
        
        print(f"\n✓ Prediction demonstration completed successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Prediction demo failed: {e}")
        print("Models not found. Please ensure training completed successfully.")
        return False

def print_api_instructions():
    """Print instructions for using the API."""
    print_section("API Instructions")
    
    print("To start the Flask API server:")
    print("  python api/app.py")
    print("")
    print("API Endpoints:")
    print("  GET  /           - Web interface")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("  GET  /categories - Available categories")
    print("  GET  /model/info - Model information")
    print("")
    print("Example API usage:")
    print("  curl -X POST http://localhost:5000/predict \\")
    print("    -H \"Content-Type: application/json\" \\")
    print("    -d '{\"text\": \"I love playing basketball\", \"return_probabilities\": true}'")

def print_project_summary():
    """Print project summary and next steps."""
    print_header("PROJECT SUMMARY")
    
    print("✓ Text Classification Project Completed Successfully!")
    print("")
    print("What was accomplished:")
    print("  ✓ Data loading and preprocessing (20 Newsgroups dataset)")
    print("  ✓ Exploratory data analysis and statistics")
    print("  ✓ Multiple model training and comparison")
    print("  ✓ Hyperparameter tuning")
    print("  ✓ Model evaluation with comprehensive metrics")
    print("  ✓ Visualization suite with multiple chart types")
    print("  ✓ Prediction interface for new texts")
    print("  ✓ Flask REST API with web interface")
    print("  ✓ Complete project documentation")
    print("")
    print("Files created:")
    print("  📁 src/data_preprocessing.py    - Data loading and preprocessing")
    print("  📁 src/model_training.py        - Model training and evaluation")
    print("  📁 src/prediction.py            - Prediction interface")
    print("  📁 src/visualization.py         - Visualization suite")
    print("  📁 api/app.py                   - Flask REST API")
    print("  📁 models/                      - Trained models and preprocessors")
    print("  📁 visualizations/              - Generated plots and charts")
    print("  📁 README.md                    - Project documentation")
    print("  📁 requirements.txt             - Dependencies")
    print("")
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run the complete pipeline: python main.py")
    print("  3. Start the API server: python api/app.py")
    print("  4. Test individual components as needed")
    print("  5. Customize for your specific use case")
    print("")
    print("🚀 Ready for deployment and further development!")

def main():
    """Main execution function."""
    start_time = time.time()
    
    print_header("AI TEXT CLASSIFICATION PROJECT")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    print_section("Setup")
    create_directories()
    
    try:
        # Step 1: Data Preprocessing
        print_header("STEP 1: DATA PREPROCESSING")
        preprocessing_success = run_data_preprocessing()
        
        if not preprocessing_success:
            print("✗ Preprocessing failed. Stopping pipeline.")
            return
        
        # Step 2: Model Training
        print_header("STEP 2: MODEL TRAINING")
        training_results = run_model_training()
        
        if not training_results:
            print("✗ Model training failed. Stopping pipeline.")
            return
        
        # Step 3: Visualizations
        print_header("STEP 3: VISUALIZATIONS")
        visualization_success = run_visualization()
        
        # Step 4: Prediction Demo
        print_header("STEP 4: PREDICTION DEMO")
        prediction_success = run_prediction_demo()
        
        # API Instructions
        print_api_instructions()
        
        # Final Summary
        print_project_summary()
        
        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()