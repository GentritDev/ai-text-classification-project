"""
Visualization Module for Text Classification
This module creates various visualizations for data analysis and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import TextPreprocessor
from model_training import ModelTrainer

class TextClassificationVisualizer:
    """
    A class for creating visualizations for text classification analysis.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size for plots
        """
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        plt.style.use('seaborn-v0_8')  # Use seaborn style for better-looking plots
    
    def plot_category_distribution(self, target, target_names, title="Category Distribution"):
        """
        Plot the distribution of categories in the dataset.
        
        Args:
            target (array): Target labels
            target_names (list): Names of target categories
            title (str): Plot title
        """
        # Count categories
        category_counts = Counter(target)
        categories = [target_names[i] for i in sorted(category_counts.keys())]
        counts = [category_counts[i] for i in sorted(category_counts.keys())]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        bars = ax1.bar(range(len(categories)), counts, color=self.colors[:len(categories)])
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title(f'{title} - Bar Plot')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=self.colors[:len(categories)])
        ax2.set_title(f'{title} - Pie Chart')
        
        plt.tight_layout()
        plt.savefig('visualizations/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_text_length_distribution(self, texts, title="Text Length Distribution"):
        """
        Plot the distribution of text lengths.
        
        Args:
            texts (list): List of texts
            title (str): Plot title
        """
        # Calculate text lengths (in words)
        text_lengths = [len(text.split()) for text in texts]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(text_lengths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Text Length (words)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.axvline(np.mean(text_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(text_lengths):.1f}')
        ax1.axvline(np.median(text_lengths), color='green', linestyle='--', 
                   label=f'Median: {np.median(text_lengths):.1f}')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(text_lengths, vert=True)
        ax2.set_ylabel('Text Length (words)')
        ax2.set_title(f'{title} - Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"Text Length Statistics:")
        print(f"  Mean: {np.mean(text_lengths):.1f} words")
        print(f"  Median: {np.median(text_lengths):.1f} words")
        print(f"  Min: {np.min(text_lengths)} words")
        print(f"  Max: {np.max(text_lengths)} words")
        print(f"  Std: {np.std(text_lengths):.1f} words")
    
    def create_word_cloud(self, texts, title="Word Cloud", max_words=100):
        """
        Create a word cloud from texts.
        
        Args:
            texts (list): List of texts
            title (str): Plot title
            max_words (int): Maximum number of words in the cloud
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            random_state=42
        ).generate(combined_text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/word_cloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_most_frequent_words(self, texts, top_n=20, title="Most Frequent Words"):
        """
        Plot the most frequent words.
        
        Args:
            texts (list): List of texts
            top_n (int): Number of top words to show
            title (str): Plot title
        """
        # Get word frequencies
        preprocessor = TextPreprocessor()
        preprocessed_texts = preprocessor.preprocess_texts(texts)
        frequent_words = preprocessor.get_most_frequent_words(preprocessed_texts, top_n)
        
        words, frequencies = zip(*frequent_words)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), frequencies, color='lightcoral')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.title(title)
        plt.yticks(range(len(words)), words)
        plt.gca().invert_yaxis()  # Highest frequency at top
        
        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            plt.text(bar.get_width() + max(frequencies)*0.01, bar.get_y() + bar.get_height()/2,
                    str(freq), ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('visualizations/frequent_words.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names, title="Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            target_names (list): Names of target categories
            title (str): Plot title
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and print accuracy per class
        print("\nPer-class Accuracy:")
        for i, class_name in enumerate(target_names):
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  {class_name}: {class_accuracy:.3f}")
    
    def plot_model_comparison(self, model_scores, title="Model Comparison"):
        """
        Plot comparison of different models.
        
        Args:
            model_scores (dict): Dictionary of model names and their scores
            title (str): Plot title
        """
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, scores, color=self.colors[:len(models)])
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line for best score
        best_score = max(scores)
        plt.axhline(y=best_score, color='red', linestyle='--', alpha=0.7,
                   label=f'Best Score: {best_score:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, title="Feature Importance"):
        """
        Plot feature importance for models that support it.
        
        Args:
            model: Trained model
            feature_names (array): Names of features
            top_n (int): Number of top features to show
            title (str): Plot title
        """
        try:
            # Get feature importance or coefficients
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                print(f"Model does not support feature importance visualization")
                return
            
            # Get top features
            top_indices = np.argsort(importance)[-top_n:]
            top_features = feature_names[top_indices]
            top_importance = importance[top_indices]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_features)), top_importance, color='lightgreen')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title(title)
            plt.yticks(range(len(top_features)), top_features)
            
            # Add importance labels
            for i, (bar, imp) in enumerate(zip(bars, top_importance)):
                plt.text(bar.get_width() + max(top_importance)*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{imp:.4f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    def create_comprehensive_report(self, categories=None):
        """
        Create a comprehensive visualization report.
        
        Args:
            categories: List of categories to analyze (None for all)
        """
        print("Creating comprehensive visualization report...")
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Load data
        preprocessor = TextPreprocessor()
        dataset = preprocessor.load_data(categories=categories)
        
        print("1. Creating category distribution plot...")
        self.plot_category_distribution(dataset['target'], dataset['target_names'])
        
        print("2. Creating text length distribution plot...")
        self.plot_text_length_distribution(dataset['data'])
        
        print("3. Creating word cloud...")
        sample_texts = dataset['data'][:1000]  # Use sample for faster processing
        self.create_word_cloud(sample_texts)
        
        print("4. Creating frequent words plot...")
        self.plot_most_frequent_words(sample_texts)
        
        # Load model results if available
        try:
            model_scores = joblib.load('models/model_comparison.pkl')
            print("5. Creating model comparison plot...")
            self.plot_model_comparison(model_scores)
        except FileNotFoundError:
            print("5. Model comparison data not found. Train models first.")
        
        # Create confusion matrix if models are available
        try:
            trainer = ModelTrainer()
            if hasattr(trainer, 'best_model') and trainer.best_model is not None:
                print("6. Creating confusion matrix...")
                y_test = getattr(trainer, 'y_test', None)
                if y_test is not None:
                    y_pred = trainer.best_model.predict(trainer.X_test)
                    self.plot_confusion_matrix(y_test, y_pred, trainer.target_names)
        except Exception as e:
            print(f"6. Could not create confusion matrix: {e}")
        
        print("Comprehensive visualization report completed!")
        print("All visualizations saved in the 'visualizations' directory.")

def main():
    """Main function to create all visualizations."""
    visualizer = TextClassificationVisualizer()
    
    # Use subset of categories for faster processing
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    
    print("Text Classification Visualization Suite")
    print("="*50)
    print("1. Create comprehensive report")
    print("2. Individual visualizations")
    
    choice = input("Choose an option (1-2): ").strip()
    
    if choice == '1':
        visualizer.create_comprehensive_report(categories=categories)
    
    elif choice == '2':
        print("\nIndividual Visualizations:")
        print("a. Category distribution")
        print("b. Text length distribution")
        print("c. Word cloud")
        print("d. Most frequent words")
        print("e. Model comparison (requires trained models)")
        
        sub_choice = input("Choose visualization (a-e): ").strip().lower()
        
        # Load data for individual visualizations
        preprocessor = TextPreprocessor()
        dataset = preprocessor.load_data(categories=categories)
        
        os.makedirs('visualizations', exist_ok=True)
        
        if sub_choice == 'a':
            visualizer.plot_category_distribution(dataset['target'], dataset['target_names'])
        elif sub_choice == 'b':
            visualizer.plot_text_length_distribution(dataset['data'])
        elif sub_choice == 'c':
            visualizer.create_word_cloud(dataset['data'][:1000])
        elif sub_choice == 'd':
            visualizer.plot_most_frequent_words(dataset['data'][:1000])
        elif sub_choice == 'e':
            try:
                model_scores = joblib.load('models/model_comparison.pkl')
                visualizer.plot_model_comparison(model_scores)
            except FileNotFoundError:
                print("Model comparison data not found. Please train models first.")
        else:
            print("Invalid choice!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()