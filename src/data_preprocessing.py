"""
Data Preprocessing Module for Text Classification
This module handles loading, cleaning, and preprocessing of text data.
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import joblib
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for the 20 Newsgroups dataset.
    """
    
    def __init__(self):
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = None
        self.label_names = None
        
    def download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            
    def load_data(self, categories=None, subset='all', remove_headers_footers=True):
        """
        Load the 20 Newsgroups dataset.
        
        Args:
            categories: List of categories to load (None for all)
            subset: 'train', 'test', or 'all'
            remove_headers_footers: Whether to remove headers and footers
            
        Returns:
            dict: Dataset containing data, target, target_names, etc.
        """
        print("Loading 20 Newsgroups dataset...")
        
        # Load training data
        train_data = fetch_20newsgroups(
            categories=categories,
            subset='train',
            remove=('headers', 'footers', 'quotes') if remove_headers_footers else (),
            random_state=42
        )
        
        # Load test data
        test_data = fetch_20newsgroups(
            categories=categories,
            subset='test',
            remove=('headers', 'footers', 'quotes') if remove_headers_footers else (),
            random_state=42
        )
        
        self.label_names = train_data.target_names
        
        if subset == 'train':
            return train_data
        elif subset == 'test':
            return test_data
        else:
            # Combine train and test data
            combined_data = {
                'data': train_data.data + test_data.data,
                'target': np.concatenate([train_data.target, test_data.target]),
                'target_names': train_data.target_names,
                'filenames': np.concatenate([train_data.filenames, test_data.filenames])
            }
            return combined_data

    def clean_text(self, text):
        """
        Clean and preprocess a single text document.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text, remove_stopwords=True, apply_stemming=False):
        """
        Tokenize text and apply filtering.
        
        Args:
            text (str): Text to tokenize
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            
        Returns:
            list: List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def preprocess_texts(self, texts, remove_stopwords=True, apply_stemming=False):
        """
        Preprocess a list of texts.
        
        Args:
            texts (list): List of raw texts
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            
        Returns:
            list: List of preprocessed texts
        """
        preprocessed = []
        
        print("Preprocessing texts...")
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(texts)} texts")
            
            # Clean text
            cleaned = self.clean_text(text)
            
            # Tokenize and filter
            tokens = self.tokenize_and_filter(cleaned, remove_stopwords, apply_stemming)
            
            # Join tokens back to string
            preprocessed_text = ' '.join(tokens)
            preprocessed.append(preprocessed_text)
        
        print(f"Preprocessing complete. Processed {len(texts)} texts.")
        return preprocessed
    
    def create_tfidf_features(self, texts, max_features=10000, ngram_range=(1, 2)):
        """
        Create TF-IDF features from texts.
        
        Args:
            texts (list): List of preprocessed texts
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range for feature extraction
            
        Returns:
            scipy.sparse matrix: TF-IDF feature matrix
        """
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            stop_words='english'
        )
        
        features = self.vectorizer.fit_transform(texts)
        
        print(f"Created TF-IDF features: {features.shape}")
        return features
    
    def get_dataset_statistics(self, data, target, target_names):
        """
        Generate basic statistics about the dataset.
        
        Args:
            data (list): List of texts
            target (array): Target labels
            target_names (list): List of target names
            
        Returns:
            dict: Dictionary containing various statistics
        """
        stats = {}
        
        # Basic dataset info
        stats['total_samples'] = len(data)
        stats['num_categories'] = len(target_names)
        stats['categories'] = target_names
        
        # Category distribution
        category_counts = Counter(target)
        stats['category_distribution'] = {
            target_names[i]: count for i, count in category_counts.items()
        }
        
        # Text length statistics
        text_lengths = [len(text.split()) for text in data]
        stats['avg_text_length'] = np.mean(text_lengths)
        stats['median_text_length'] = np.median(text_lengths)
        stats['min_text_length'] = np.min(text_lengths)
        stats['max_text_length'] = np.max(text_lengths)
        
        return stats
    
    def get_most_frequent_words(self, texts, top_n=20):
        """
        Get the most frequent words across all texts.
        
        Args:
            texts (list): List of preprocessed texts
            top_n (int): Number of top words to return
            
        Returns:
            list: List of (word, frequency) tuples
        """
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Get word frequencies
        words = all_text.split()
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor (vectorizer) to disk."""
        if self.vectorizer is not None:
            joblib.dump({
                'vectorizer': self.vectorizer,
                'label_names': self.label_names
            }, filepath)
            print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load the preprocessor from disk."""
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.label_names = data['label_names']
        print(f"Preprocessor loaded from {filepath}")

def main():
    """Main function to demonstrate preprocessing pipeline."""
    preprocessor = TextPreprocessor()
    
    # Load data (using a subset for faster demonstration)
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'rec.sport.hockey']
    dataset = preprocessor.load_data(categories=categories)
    
    print(f"\nDataset loaded with {len(dataset['data'])} samples")
    print(f"Categories: {dataset['target_names']}")
    
    # Get basic statistics
    stats = preprocessor.get_dataset_statistics(
        dataset['data'], dataset['target'], dataset['target_names']
    )
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of categories: {stats['num_categories']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} words")
    print(f"Category distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    # Preprocess texts
    preprocessed_texts = preprocessor.preprocess_texts(dataset['data'][:1000])  # Sample for demo
    
    # Get most frequent words
    frequent_words = preprocessor.get_most_frequent_words(preprocessed_texts, top_n=15)
    print(f"\nMost frequent words:")
    for word, freq in frequent_words:
        print(f"  {word}: {freq}")
    
    # Create TF-IDF features
    features = preprocessor.create_tfidf_features(preprocessed_texts)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print(f"\nPreprocessing pipeline completed successfully!")
    print(f"Feature matrix shape: {features.shape}")

if __name__ == "__main__":
    main()