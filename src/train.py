from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def train_models(X, labels):
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42)

    # 2. Debug: print what classes we're training on
    print("✅ Training on labels:", y_train.unique())

    # 3. Scale TF-IDF features (very important for linear models)
    scaler = StandardScaler(with_mean=False)  # with_mean=False is required for sparse matrices
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=1000, class_weight='balanced')
    pac.fit(X_train, y_train)

    # 5. Train Support Vector Machine
    svm_model = svm.LinearSVC(class_weight='balanced')
    svm_model.fit(X_train, y_train)

    return pac, svm_model, X_test, y_test
