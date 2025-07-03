# app.py
from src.preprocess import load_data, vectorize_text
from src.train import train_models
from src.evaluate import evaluate_model

def main():
    print("📥 Loading data...")
    text, labels = load_data("data/news.csv")

    print("🧹 Vectorizing text...")
    X, tfidf = vectorize_text(text)

    print("🎯 Training models...")
    pac, svm_model, X_test, y_test = train_models(X, labels)

    print("📊 Evaluating models...")
    evaluate_model(pac, X_test, y_test, "Passive Aggressive Classifier")
    evaluate_model(svm_model, X_test, y_test, "SVM")

if __name__ == "__main__":
    main()
