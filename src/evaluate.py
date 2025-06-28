from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)

    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Use 'Fake' as the positive class — this is critical!
    print("F1 Score (Fake as positive):", f1_score(y_test, predictions, pos_label='Fake'))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Debug: Show prediction samples
    for true_label, pred in list(zip(y_test, predictions))[:5]:
        print(f"  True: {true_label}, Predicted: {pred}")

    from collections import Counter
    print("🧠 Predicted label distribution:", Counter(predictions))

