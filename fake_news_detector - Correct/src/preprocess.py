import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Capitalize labels to standardize (important!)
    df['label'] = df['label'].str.strip().str.capitalize()


    # Debug: See how labels are distributed
    print("✅ Label counts in dataset:")
    print(df['label'].value_counts())

    return df['text'], df['label']

def vectorize_text(text):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Safe configuration for small datasets
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)


    print("✅ Fitting TF-IDF on text...")
    X = tfidf.fit_transform(text)

    print("✅ Shape of TF-IDF matrix:", X.shape)
    return X, tfidf

