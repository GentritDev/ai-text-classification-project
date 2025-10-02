"""
Prediction Module for Text Classification
This module handles making predictions on new text data.
"""

import joblib
import numpy as np
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import TextPreprocessor

class TextClassificationPredictor:
    """
    A class for making predictions using trained text classification models.
    """
    
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file
        """
        self.model = None
        self.model_name = None
        self.preprocessor = None
        self.target_names = None
        self.model_accuracy = None
        
        # Default paths
        if model_path is None:
            model_path = 'models/best_model.pkl'
        if preprocessor_path is None:
            preprocessor_path = 'models/preprocessor.pkl'
        
        # Load model and preprocessor
        self.load_model(model_path)
        self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path):
        """Load the trained model from disk."""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_accuracy = model_data['accuracy']
            self.target_names = model_data['target_names']
            print(f"Loaded model: {self.model_name} (Accuracy: {self.model_accuracy:.4f})")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def load_preprocessor(self, preprocessor_path):
        """Load the preprocessor from disk."""
        if os.path.exists(preprocessor_path):
            self.preprocessor = TextPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            print("Preprocessor loaded successfully")
        else:
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    def preprocess_text(self, text):
        """
        Preprocess a single text for prediction.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            scipy.sparse matrix: Preprocessed text features
        """
        # Clean and preprocess the text
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize_and_filter(cleaned_text, remove_stopwords=True)
        preprocessed_text = ' '.join(tokens)
        
        # Transform using the fitted vectorizer
        features = self.preprocessor.vectorizer.transform([preprocessed_text])
        
        return features
    
    def predict_single(self, text, return_probabilities=False):
        """
        Make a prediction for a single text.
        
        Args:
            text (str): Text to classify
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before making predictions")
        
        # Preprocess the text
        features = self.preprocess_text(text)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        predicted_category = self.target_names[prediction]
        
        result = {
            'text': text,
            'predicted_class_id': int(prediction),
            'predicted_category': predicted_category,
            'model_used': self.model_name,
            'model_accuracy': self.model_accuracy
        }
        
        # Add probabilities if requested and model supports it
        if return_probabilities:
            try:
                probabilities = self.model.predict_proba(features)[0]
                # Create a dictionary of category: probability
                prob_dict = {
                    self.target_names[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
                # Sort by probability (descending)
                prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                result['probabilities'] = prob_dict
                result['confidence'] = float(max(probabilities))
            except AttributeError:
                print(f"Model {self.model_name} does not support probability predictions")
                result['confidence'] = None
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Make predictions for multiple texts.
        
        Args:
            texts (list): List of texts to classify
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        print(f"Making predictions for {len(texts)} texts...")
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} texts")
            
            result = self.predict_single(text, return_probabilities)
            results.append(result)
        
        print("Batch prediction completed")
        return results
    
    def interactive_prediction(self):
        """
        Interactive prediction interface for testing.
        """
        print(f"\n{'='*60}")
        print("INTERACTIVE TEXT CLASSIFICATION")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Accuracy: {self.model_accuracy:.4f}")
        print(f"Categories: {', '.join(self.target_names)}")
        print("\nInstructions:")
        print("- Enter text to classify (or 'quit' to exit)")
        print("- Type 'examples' to see sample texts")
        print("- Type 'categories' to see all available categories")
        print(f"{'='*60}")
        
        while True:
            try:
                user_input = input("\nEnter text to classify: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'examples':
                    print("\nExample texts to try:")
                    examples = [
                        "I love playing basketball and watching NBA games",
                        "The graphics card is overheating and causing display issues",
                        "Prayer and meditation help me find inner peace",
                        "The patient shows symptoms of acute bronchitis",
                        "God loves everyone regardless of their beliefs"
                    ]
                    for i, example in enumerate(examples, 1):
                        print(f"{i}. {example}")
                    continue
                
                elif user_input.lower() == 'categories':
                    print(f"\nAvailable categories ({len(self.target_names)}):")
                    for i, category in enumerate(self.target_names, 1):
                        print(f"{i:2d}. {category}")
                    continue
                
                elif not user_input:
                    print("Please enter some text to classify.")
                    continue
                
                # Make prediction
                result = self.predict_single(user_input, return_probabilities=True)
                
                # Display results
                print(f"\n{'='*40}")
                print("PREDICTION RESULTS")
                print(f"{'='*40}")
                print(f"Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
                print(f"Predicted Category: {result['predicted_category']}")
                
                if result.get('confidence'):
                    print(f"Confidence: {result['confidence']:.4f}")
                
                if result.get('probabilities'):
                    print(f"\nTop 3 predictions:")
                    for i, (category, prob) in enumerate(list(result['probabilities'].items())[:3], 1):
                        print(f"{i}. {category}: {prob:.4f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def demonstrate_predictions():
    """Demonstrate the prediction functionality with sample texts."""
    predictor = TextClassificationPredictor()
    
    # Sample texts for different categories
    sample_texts = [
        "I love playing hockey and watching NHL games every weekend",
        "The new graphics processing unit is incredible for gaming",
        "Prayer and faith have always been important in my life",
        "The patient needs immediate medical attention for chest pain",
        "Atheism is simply the lack of belief in deities"
    ]
    
    print("Demonstrating predictions on sample texts:")
    print("="*50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text[:80]}{'...' if len(text) > 80 else ''}")
        result = predictor.predict_single(text, return_probabilities=True)
        print(f"Prediction: {result['predicted_category']}")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']:.4f}")

def main():
    """Main function with options for different prediction modes."""
    try:
        print("Text Classification Predictor")
        print("="*40)
        print("1. Interactive prediction")
        print("2. Demo with sample texts")
        print("3. Batch prediction from file")
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == '1':
            predictor = TextClassificationPredictor()
            predictor.interactive_prediction()
        
        elif choice == '2':
            demonstrate_predictions()
        
        elif choice == '3':
            file_path = input("Enter path to text file (one text per line): ").strip()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                
                predictor = TextClassificationPredictor()
                results = predictor.predict_batch(texts, return_probabilities=True)
                
                # Save results
                output_file = file_path.replace('.txt', '_predictions.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"Text: {result['text']}\n")
                        f.write(f"Prediction: {result['predicted_category']}\n")
                        if result.get('confidence'):
                            f.write(f"Confidence: {result['confidence']:.4f}\n")
                        f.write("-" * 40 + "\n")
                
                print(f"Results saved to {output_file}")
            else:
                print("File not found!")
        
        else:
            print("Invalid choice!")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained models available.")
        print("Run 'python src/model_training.py' first to train the models.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()