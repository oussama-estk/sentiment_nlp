# # =================================================================
# # app.py - Flask API for Render
# # =================================================================

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

# # --- Initialize App ---
# app = Flask(__name__)
# CORS(app)

# # --- Load NLTK Resources (Required for Render) ---
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)  # <--- THIS IS THE NEW LINE YOU NEED
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # --- Load Model ---
# MODEL_PATH = 'sentiment_model_pipeline.joblib'
# try:
#     pipeline = joblib.load(MODEL_PATH)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     pipeline = None

# # --- Helper Function: Text Cleaning ---
# # Must match the logic used in training exactly
# def advanced_clean_text(text):
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))

#     text = text.lower()
#     text = re.sub(r'<.*?>', ' ', text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     tokens = word_tokenize(text)
#     cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
#     return " ".join(cleaned_tokens)

# # --- Routes ---

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({'message': 'Sentiment Analysis API is Running!'})

# @app.route('/predict', methods=['POST'])
# def predict():
#     if not pipeline:
#         return jsonify({'error': 'Model not loaded'}), 500

#     data = request.get_json()
    
#     if not data or 'text' not in data:
#         return jsonify({'error': 'Field "text" is required.'}), 400

#     raw_text = data['text']
#     cleaned_text = advanced_clean_text(raw_text)

#     # Prediction
#     prediction = pipeline.predict([cleaned_text])[0]
#     probabilities = pipeline.predict_proba([cleaned_text])[0]

#     result = {
#         'original_text': raw_text,
#         'sentiment': 'positive' if prediction == 1 else 'negative',
#         'confidence': float(probabilities[prediction])
#     }

#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)

# =================================================================
# app.py - Fixed for NLTK on Railway
# =================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Initialize App ---
app = Flask(__name__)
CORS(app)

# --- FORCE NLTK CONFIGURATION ---
# 1. Define a specific local folder inside the app for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# 2. Create the directory if it does not exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# 3. Tell NLTK to look in this folder FIRST
nltk.data.path.append(nltk_data_dir)

# 4. Download resources directly to this folder
print(f"Downloading NLTK data to: {nltk_data_dir}...")
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True) # CRITICAL FIX
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
print("NLTK data downloaded successfully.")

# --- Load Model ---
MODEL_PATH = 'sentiment_model_pipeline.joblib'
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

# --- Helper Function: Text Cleaning ---
def advanced_clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize using the downloaded data
    tokens = word_tokenize(text)
    
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Sentiment Analysis API is Running!'})

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Field "text" is required.'}), 400

    try:
        raw_text = data['text']
        cleaned_text = advanced_clean_text(raw_text)

        # Prediction
        prediction = pipeline.predict([cleaned_text])[0]
        probabilities = pipeline.predict_proba([cleaned_text])[0]

        result = {
            'original_text': raw_text,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(probabilities[prediction])
        }
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
