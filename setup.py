"""
Setup Script for Text Classification Project
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is suitable."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        return False

def install_requirements():
    """Install required packages."""
    if os.path.exists('requirements.txt'):
        print("Installing requirements from requirements.txt...")
        return run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python packages"
        )
    else:
        print("requirements.txt not found. Installing packages manually...")
        packages = [
            "scikit-learn==1.3.0",
            "pandas==2.0.3",
            "numpy==1.24.3",
            "nltk==3.8.1",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "wordcloud==1.9.2",
            "Flask==2.3.2",
            "Flask-CORS==4.0.0",
            "jupyter==1.0.0",
            "joblib==1.3.2"
        ]
        
        for package in packages:
            if not run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package}"
            ):
                return False
        return True

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'visualizations', 'data']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    required_modules = [
        'numpy', 'pandas', 'sklearn', 'nltk', 'matplotlib', 
        'seaborn', 'wordcloud', 'flask', 'joblib'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    print("✓ All required modules can be imported")
    return True

def main():
    """Main setup function."""
    print("Text Classification Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\\nCreating directories...")
    if not create_directories():
        print("Failed to create directories")
        return
    
    # Install requirements
    print("\\nInstalling requirements...")
    if not install_requirements():
        print("Failed to install requirements")
        return
    
    # Download NLTK data
    print("\\nDownloading NLTK data...")
    if not download_nltk_data():
        print("Failed to download NLTK data")
        return
    
    # Test imports
    print("\\nTesting imports...")
    if not test_imports():
        print("Some imports failed. Please check the installation.")
        return
    
    print("\\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\\nNext steps:")
    print("1. Run the complete pipeline: python main.py")
    print("2. Start the API server: python api/app.py")
    print("3. Open Jupyter notebook: jupyter notebook notebooks/")
    print("4. Run individual components as needed")
    print("\\nProject is ready to use! 🚀")

if __name__ == "__main__":
    main()