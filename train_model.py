# =================================================================
# train_model.py
# Purpose: Preprocess text, train Logistic Regression, save pipeline.
# =================================================================

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- Download NLTK Resources ---
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Configuration ---
DATA_FILE = 'data.json'
MODEL_FILE = 'sentiment_model_pipeline.joblib'

def advanced_clean_text(text):
    """
    Cleans raw text by removing HTML, URLs, special characters,
    and performing lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = word_tokenize(text)
    # Lemmatize and remove stopwords
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(cleaned_tokens)

def train():
    # 1. Load Data
    print(f"\nLoading data from {DATA_FILE}...")
    try:
        df = pd.read_json(DATA_FILE)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 2. Preprocessing
    print("Preprocessing text (this may take a moment)...")
    df['cleaned_review'] = df['review'].apply(advanced_clean_text)
    
    # Encode Target: positive -> 1, negative -> 0
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    X = df['cleaned_review']
    y = df['label']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # 4. Build Pipeline
    # Using TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
    ])

    # 5. Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 7. Save Model
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved successfully to '{MODEL_FILE}'")

if __name__ == "__main__":
    train()