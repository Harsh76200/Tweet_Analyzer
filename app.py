from flask import Flask, request, render_template, jsonify
import pickle
import re
import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.sparse as sp
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load model and preprocessing tools
try:
    with open("gradient_boosting_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        print("===========Model loaded successfully")

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        print("============Vectorizer loaded successfully")
        
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        print("============Scaler loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define numerical features
numerical_features = [
    'tweet_length', 'word_count', 'avg_word_length',
    'neg_score', 'neu_score', 'pos_score', 'compound_score',
    'exclamation_count', 'question_count',
    'retweets', 'likes', 'engagement_ratio'
]

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("============SpaCy model loaded successfully")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    # Fallback to basic processing if spaCy not available
    nlp = None

# Clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Use spaCy for lemmatization if available
    if nlp:
        doc = nlp(text)  # Process text using spaCy
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]  # Remove stopwords and short tokens
        return " ".join(tokens)
    else:
        # Simple fallback if spaCy not available
        return text

def predict_candidate(tweet_text):
    """Complete prediction pipeline for a new tweet"""
    # Preprocess the text
    cleaned_text = preprocess_text(tweet_text)
    
    # Get TF-IDF features
    text_features = vectorizer.transform([cleaned_text]).toarray()
    
    # Get sentiment and other text features
    sentiment_scores_dict = analyzer.polarity_scores(cleaned_text)
    
    # Create numerical features
    numerical_data = {
        'tweet_length': len(cleaned_text),
        'word_count': len(cleaned_text.split()),
        'avg_word_length': np.mean([len(word) for word in cleaned_text.split()] or [0]),
        'neg_score': sentiment_scores_dict['neg'],
        'neu_score': sentiment_scores_dict['neu'],
        'pos_score': sentiment_scores_dict['pos'],
        'compound_score': sentiment_scores_dict['compound'],
        'exclamation_count': tweet_text.count('!'),
        'question_count': tweet_text.count('?'),
        'retweets': 0,  # Default value for prediction
        'likes': 0,     # Default value for prediction
        'engagement_ratio': 0  # Default value for prediction
    }
    
    # Convert to DataFrame and scale
    numerical_df = pd.DataFrame([numerical_data])
    numerical_scaled = scaler.transform(numerical_df)
    
    # Check dimensions
    expected_text_features = model.n_features_in_ - len(numerical_features)
    
    # Align text features if necessary
    if text_features.shape[1] != expected_text_features:
        # Pad or truncate as needed
        aligned_text_features = np.zeros((1, expected_text_features))
        min_cols = min(text_features.shape[1], expected_text_features)
        aligned_text_features[0, :min_cols] = text_features[0, :min_cols]
        text_features = aligned_text_features
    
    # Combine features
    all_features = np.hstack((numerical_scaled, text_features))
    
    # Make prediction
    prediction = model.predict(all_features)[0]
    probabilities = model.predict_proba(all_features)[0]
    
    # Get class labels and probabilities
    class_labels = model.classes_
    prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
    
    return prediction, prob_dict, sentiment_scores_dict, numerical_data

def get_word_importance(text):
    """Extract top words and their importance from the text"""
    # Clean and tokenize the text
    cleaned = preprocess_text(text)
    words = cleaned.split()
    
    # Get word counts
    word_counts = Counter(words)
    
    # Return top 5 words or all if less than 5
    top_words = word_counts.most_common(5)
    return top_words

# Get relevant sentiment statistics data
def get_sentiment_data():
    candidates = ["Kamala Harris", "Donald Trump", "Jill Stein", "Robert Kennedy", "Chase Oliver"]
    sentiment_data = {
        "sentiment_counts": {"positive": 45, "neutral": 30, "negative": 25},
        "candidate_sentiment": {
            "Kamala Harris": {"positive": 40, "neutral": 30, "negative": 30},
            "Donald Trump": {"positive": 35, "neutral": 25, "negative": 40},
            "Jill Stein": {"positive": 55, "neutral": 25, "negative": 20},
            "Robert Kennedy": {"positive": 30, "neutral": 45, "negative": 25},
            "Chase Oliver": {"positive": 38, "neutral": 42, "negative": 20}
        }
    }
    return sentiment_data

@app.route("/")
def index():
    sentiment_data = get_sentiment_data()
    return render_template("index.html", sentiment_data=sentiment_data)

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request is AJAX or form submission
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        data = request.get_json()
        text = data.get("tweet", "")
    else:
        text = request.form.get("tweet", "")
    
    if not text.strip():
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": "Please enter some text."})
        return render_template("index.html", error="Please enter some text.")

    # Run prediction pipeline
    candidate_prediction, confidence_scores, sentiment_scores, feature_data = predict_candidate(text)
    
    # Determine sentiment category
    sentiment = "positive" if sentiment_scores['compound'] > 0.05 else "negative" if sentiment_scores['compound'] < -0.05 else "neutral"
    
    # Get important words
    important_words = get_word_importance(text)
    
    # Format confidence scores for display
    confidence_data = [{"candidate": cand, "score": score} for cand, score in confidence_scores.items()]
    confidence_data.sort(key=lambda x: x["score"], reverse=True)
    
    # Get sentiment statistics data
    sentiment_data = get_sentiment_data()
    
    # Prepare result dictionary
    result = {
        "tweet": text,
        "sentiment": sentiment,
        "sentiment_details": {
            "compound": round(sentiment_scores['compound'], 2),
            "positive": round(sentiment_scores['pos'], 2),
            "negative": round(sentiment_scores['neg'], 2),
            "neutral": round(sentiment_scores['neu'], 2)
        },
        "candidate_prediction": candidate_prediction,
        "confidence_scores": confidence_data,
        "important_words": important_words,
        "sentiment_data": sentiment_data,
        "feature_data": {k: round(float(v), 2) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in feature_data.items()}
    }
    
    # Return JSON if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(result)
    
    # Otherwise render the template
    return render_template("index.html", **result)

@app.route("/api/sentiment_data")
def get_sentiment_data_api():
    """API endpoint to get sentiment data for visualizations"""
    return jsonify(get_sentiment_data())

@app.route("/api/feature_importance")
def get_feature_importance():
    """API endpoint to get feature importance data"""
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = numerical_features + [f"word_{i}" for i in range(model.n_features_in_ - len(numerical_features))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        top_features = [{"feature": feature_names[i], "importance": float(importances[i])} 
                        for i in indices[:20]]  # Get top 20
        
        return jsonify({"feature_importance": top_features})
    else:
        return jsonify({"error": "Feature importance not available for this model type"})

if __name__ == "__main__":
    print("=============Server started at http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=True)