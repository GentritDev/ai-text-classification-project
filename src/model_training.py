"""
Model Training Module for Text Classification
This module handles training and evaluation of multiple ML models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import os
import time
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import TextPreprocessor

class ModelTrainer:
    """
    A comprehensive model training class for text classification.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = TextPreprocessor()
        
    def initialize_models(self):
        """Initialize different ML models for comparison."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'Multinomial Naive Bayes': MultinomialNB(
                alpha=1.0
            ),
            'Linear SVM': LinearSVC(
                random_state=42,
                C=1.0,
                max_iter=2000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=20
            )
        }
        
        print(f"Initialized {len(self.models)} models for training:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def prepare_data(self, categories=None, test_size=0.2, random_state=42):
        """
        Prepare data for training.
        
        Args:
            categories: List of categories to use (None for all)
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
        """
        print("Loading and preprocessing data...")
        
        # Load dataset
        dataset = self.preprocessor.load_data(categories=categories)
        
        # Preprocess texts
        preprocessed_texts = self.preprocessor.preprocess_texts(
            dataset['data'], 
            remove_stopwords=True, 
            apply_stemming=False
        )
        
        # Create TF-IDF features
        X = self.preprocessor.create_tfidf_features(preprocessed_texts)
        y = dataset['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data prepared:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test, self.preprocessor.vectorizer.get_feature_names_out(), dataset['target_names']
    
    def train_model(self, model_name, model, X_train, y_train):
        """
        Train a single model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            trained model
        """
        print(f"Training {model_name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        
        return model
    
    def evaluate_model(self, model, model_name, X_test, y_test, target_names):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            target_names: Names of target classes
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Create evaluation results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, target_names=target_names),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def perform_cross_validation(self, model, model_name, X, y, cv=5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            model_name: Name of the model
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Performing {cv}-fold cross-validation for {model_name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print(f"  CV Accuracy: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score'] * 2:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """
        Perform hyperparameter tuning for specific models.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best model with tuned hyperparameters
        """
        print(f"Tuning hyperparameters for {model_name}...")
        
        if model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
            
        elif model_name == 'Linear SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'loss': ['hinge', 'squared_hinge']
            }
            model = LinearSVC(random_state=42, max_iter=2000)
            
        elif model_name == 'Multinomial Naive Bayes':
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            }
            model = MultinomialNB()
            
        else:
            print(f"  No hyperparameter tuning implemented for {model_name}")
            return self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, categories=None, perform_tuning=False):
        """
        Train all models and compare their performance.
        
        Args:
            categories: List of categories to use (None for all)
            perform_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            dict: Results for all models
        """
        # Initialize models
        self.initialize_models()
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names, target_names = self.prepare_data(categories)
        
        # Store data for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_names = target_names
        
        # Train and evaluate all models
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Processing {model_name}")
            print(f"{'='*50}")
            
            # Hyperparameter tuning (optional)
            if perform_tuning:
                model = self.hyperparameter_tuning(model_name, X_train, y_train)
            
            # Train model
            trained_model = self.train_model(model_name, model, X_train, y_train)
            self.trained_models[model_name] = trained_model
            
            # Evaluate model
            evaluation_results = self.evaluate_model(
                trained_model, model_name, X_test, y_test, target_names
            )
            
            # Cross-validation
            cv_results = self.perform_cross_validation(
                trained_model, model_name, X_train, y_train
            )
            
            # Combine results
            model_results = {
                'model': trained_model,
                'evaluation': evaluation_results,
                'cross_validation': cv_results
            }
            
            all_results[model_name] = model_results
            self.model_scores[model_name] = evaluation_results['accuracy']
        
        # Identify best model
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\n{'='*50}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            eval_results = results['evaluation']
            print(f"{model_name:<25} {eval_results['accuracy']:<12.4f} "
                  f"{eval_results['precision']:<12.4f} {eval_results['recall']:<10.4f} "
                  f"{eval_results['f1_score']:<10.4f}")
        
        print(f"\nBest Model: {self.best_model_name} (Accuracy: {self.model_scores[self.best_model_name]:.4f})")
        
        return all_results
    
    def save_models(self, models_dir='models'):
        """Save trained models to disk."""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save all trained models
        for model_name, model in self.trained_models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            filepath = os.path.join(models_dir, filename)
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(models_dir, 'best_model.pkl')
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'accuracy': self.model_scores[self.best_model_name],
                'target_names': self.target_names
            }, best_model_path)
            print(f"Saved best model ({self.best_model_name}) to {best_model_path}")
        
        # Save model comparison results
        results_path = os.path.join(models_dir, 'model_comparison.pkl')
        joblib.dump(self.model_scores, results_path)
        print(f"Saved model comparison results to {results_path}")

def main():
    """Main function to demonstrate model training pipeline."""
    trainer = ModelTrainer()
    
    # Use a subset of categories for faster training (remove this line to use all 20 categories)
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    
    print("Starting model training pipeline...")
    print(f"Categories: {categories}")
    
    # Train all models
    results = trainer.train_all_models(categories=categories, perform_tuning=True)
    
    # Save models
    trainer.save_models()
    
    print("\nModel training pipeline completed successfully!")
    print(f"All models saved in the 'models' directory")
    
    # Display detailed results for best model
    best_results = results[trainer.best_model_name]
    print(f"\nDetailed results for best model ({trainer.best_model_name}):")
    print("\nClassification Report:")
    print(best_results['evaluation']['classification_report'])

if __name__ == "__main__":
    main()