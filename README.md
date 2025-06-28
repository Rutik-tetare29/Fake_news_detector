# 📰 Fake News Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*An advanced NLP-powered system for detecting fake news articles using machine learning algorithms*

</div>

---

## 🎯 Overview

In today's digital age, the spread of misinformation poses a significant threat to society. This project implements a robust fake news detection system that leverages **Natural Language Processing (NLP)** and **Machine Learning** algorithms to classify news articles as either **authentic** or **potentially fake** with high accuracy.

The system uses advanced text preprocessing techniques, TF-IDF vectorization, and multiple machine learning models to achieve reliable detection results.

---

## ✨ Key Features

- **🤖 Multi-Model Architecture**: Implements both Passive Aggressive Classifier and SVM for robust predictions
- **🔍 Advanced Text Processing**: Comprehensive preprocessing pipeline with tokenization, stopword removal, and TF-IDF vectorization
- **🌐 Interactive Web Interface**: User-friendly Streamlit application for real-time news analysis
- **📊 Model Persistence**: Trained models are saved and can be reloaded for inference
- **📈 Performance Metrics**: Detailed evaluation with accuracy, precision, recall, and F1-score
- **🔄 Scalable Pipeline**: Modular architecture for easy extension and maintenance

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.8+ | Core development |
| **ML Framework** | Scikit-learn | Model training and evaluation |
| **NLP Processing** | NLTK | Text preprocessing and tokenization |
| **Web Framework** | Streamlit | Interactive user interface |
| **Data Handling** | Pandas, NumPy | Data manipulation and analysis |
| **Model Persistence** | Joblib | Model serialization |

---

## 📁 Project Architecture

```
fake_news_detector/
├── 📂 data/                    # Dataset storage
│   ├── Fake.csv               # Fake news samples
│   ├── True.csv               # Authentic news samples
│   └── news.csv               # Combined dataset
├── 📂 model/                   # Trained model artifacts
│   ├── pac_model.pkl          # Passive Aggressive Classifier
│   └── tfidf_vectorizer.pkl   # TF-IDF Vectorizer
├── 📂 src/                     # Source code modules
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── train.py               # Model training logic
│   └── evaluate.py            # Model evaluation metrics
├── 📄 app.py                   # Main training pipeline
├── 📄 streamlit_app.py         # Web application interface
├── 📄 combine_dataset.py       # Dataset combination utility
├── 📄 notebook.ipynb          # Jupyter analysis notebook
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

---

## 🚀 Quick Start

## 📥 Download Dataset

The datasets (`news.csv`, `Fake.csv`, and `True.csv`) are hosted on Google Drive.

To download them:

```bash
pip install gdown
python download_data.py

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake_news_detector
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first time only)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

### Usage

#### 🎯 Training the Model

Run the complete training pipeline:

```bash
python app.py
```

This will:
- Load and preprocess the dataset
- Apply TF-IDF vectorization
- Train both Passive Aggressive and SVM models
- Evaluate model performance
- Save trained models for inference

#### 🌐 Web Application

Launch the interactive Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open your browser and navigate to `http://localhost:8501` to access the web interface.

#### 📓 Jupyter Analysis

For detailed analysis and experimentation:

```bash
jupyter notebook notebook.ipynb
```

---

## 🔧 Model Details

### Algorithms Implemented

1. **Passive Aggressive Classifier**
   - Online learning algorithm ideal for large-scale text classification
   - Maintains performance while being computationally efficient
   - Particularly effective for binary classification tasks

2. **Support Vector Machine (SVM)**
   - Robust performance on high-dimensional text data
   - Good generalization capabilities
   - Effective with TF-IDF feature representation

### Feature Engineering

- **Text Preprocessing**: Lowercasing, punctuation removal, tokenization
- **Stopword Removal**: Elimination of common words that don't contribute to classification
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency for feature extraction
- **Dimensionality Optimization**: Balanced feature selection for optimal performance

---

## 📊 Performance Metrics

The models are evaluated using comprehensive metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correctly predicted positive observations
- **Recall**: Ratio of correctly predicted positive observations to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction results

---

## 🔄 API Reference

### Core Functions

```python
# Data preprocessing
from src.preprocess import load_data, vectorize_text

# Model training
from src.train import train_models

# Model evaluation
from src.evaluate import evaluate_model
```

### Usage Example

```python
# Load and preprocess data
text, labels = load_data("data/news.csv")
X, tfidf = vectorize_text(text)

# Train models
pac_model, svm_model, X_test, y_test = train_models(X, labels)

# Evaluate performance
evaluate_model(pac_model, X_test, y_test, "Passive Aggressive")
```

---

## 🤝 Contributing

We welcome contributions to improve the fake news detection system! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📈 Future Enhancements

- [ ] **Deep Learning Integration**: Implement LSTM/BERT models for improved accuracy
- [ ] **Real-time Processing**: Add streaming data processing capabilities
- [ ] **Multi-language Support**: Extend detection to multiple languages
- [ ] **API Development**: Create REST API for integration with other systems
- [ ] **Enhanced Visualization**: Add more detailed analytics and visualizations
- [ ] **Ensemble Methods**: Combine multiple models for better performance

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Rutik Tetare** - *Initial work* - [GitHub](https://github.com/Rutik-tetare29)

---

## 🙏 Acknowledgments

- Dataset sources for providing training data
- Scikit-learn community for excellent machine learning tools
- Streamlit team for the intuitive web framework
- NLTK contributors for natural language processing capabilities

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/fake_news_detector/issues) page
2. Create a new issue with detailed description
3. Contact: rutiktetare@gmail.com

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for a better, more informed world

</div>


