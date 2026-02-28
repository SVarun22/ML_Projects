import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from train_models import (CombinedFeatures, NLPFeatureExtractor,
                           TextSelector, clean_text,
                           extract_nlp_features, CLICKBAIT_KWS)

import re, json, warnings
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib, nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, 'frontend', 'templates'),
    static_folder=os.path.join(BASE, 'frontend', 'static')
)
CORS(app)

vader      = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()


def load_models():
    bp  = os.path.join(BASE, 'models', 'best_model.pkl')
    aml = os.path.join(BASE, 'models', 'all_models.pkl')
    mp  = os.path.join(BASE, 'models', 'metadata.json')
    if not os.path.exists(bp):
        raise FileNotFoundError("Models not found. Run: python train_models.py first.")
    best  = joblib.load(bp)
    all_m = joblib.load(aml)
    with open(mp) as f:
        meta = json.load(f)
    return best, all_m, meta


try:
    best_model, all_models, metadata = load_models()
    print(f"✓ Models loaded. Best: {metadata['best_model']}")
except Exception as e:
    print(f"✗ {e}")
    best_model = all_models = metadata = None


def predict_with_model(bundle, raw_text):
    clf       = bundle['clf']
    feat      = bundle['feat']
    is_nb     = bundle.get('nb', False)
    threshold = bundle.get('threshold', 0.5)   # use tuned threshold

    cleaned = clean_text(raw_text)
    X = feat.transform([cleaned])
    if is_nb:
        X = sp.csr_matrix(np.abs(X.toarray()))

    prob     = clf.predict_proba(X)[0]
    raw_fake = float(prob[1])
    raw_real = float(prob[0])

    # Apply tuned threshold for label
    label    = 'Fake' if raw_fake >= threshold else 'Real'

    # Scale probabilities to 0-100 for display
    fake_pct = round(raw_fake * 100, 2)
    real_pct = round(raw_real * 100, 2)
    confidence = round(max(fake_pct, real_pct), 2)

    return fake_pct, real_pct, label, confidence


def build_nlp_breakdown(raw_text):
    t     = str(raw_text)
    words = t.split()
    try:
        sentences = sent_tokenize(t)
    except Exception:
        sentences = [t]

    cb_found   = [kw for kw in CLICKBAIT_KWS if kw in t.lower()]
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    vs         = vader.polarity_scores(t)
    blob       = TextBlob(t)

    excl_count = t.count('!')
    subj       = round(blob.sentiment.subjectivity * 100, 1)
    sentiment  = round(abs(vs['compound']) * 100, 1)
    neg_sent   = round(vs['neg'] * 100, 1)
    lex_div    = round(len(set(words)) / (len(words) + 1) * 100, 1)
    punc_abuse = round((t.count('!') + t.count('?')) / (len(t) / 100 + 1), 2)

    signals = {
        'Clickbait Keywords' : round(min(len(cb_found) / len(CLICKBAIT_KWS) * 100 * 5, 100), 1),
        'ALL-CAPS Abuse'     : round(min(len(caps_words) / (len(words) + 1) * 100 * 10, 100), 1),
        'Exclamation Marks'  : round(min(excl_count * 15, 100), 1),
        'Subjectivity'       : subj,
        'Emotional Extremity': sentiment,
        'Negative Sentiment' : round(min(neg_sent * 2, 100), 1),
        'Punctuation Abuse'  : round(min(punc_abuse * 10, 100), 1),
        'Low Lex. Diversity' : round(max(0, 60 - lex_div), 1),
    }

    return {
        'clickbait_keywords' : cb_found,
        'caps_words'         : caps_words[:10],
        'exclamation_count'  : excl_count,
        'subjectivity_pct'   : subj,
        'sentiment_extremity': sentiment,
        'neg_sentiment_pct'  : neg_sent,
        'lexical_diversity'  : lex_div,
        'word_count'         : len(words),
        'sentence_count'     : len(sentences),
        'has_numbers'        : bool(re.search(r'\d+', t)),
        'has_quotes'         : bool('"' in t or "'" in t),
        'signals'            : signals,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if best_model is None:
        return jsonify({'error': 'Models not loaded. Run train_models.py first.'}), 500

    data = request.get_json()
    text = (data or {}).get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    if len(text) < 20:
        return jsonify({'error': 'Text too short — enter at least 20 characters.'}), 400

    fake_pct, real_pct, label, conf = predict_with_model(best_model, text)
    nlp = build_nlp_breakdown(text)

    return jsonify({
        'label'     : label,
        'fake_pct'  : fake_pct,
        'real_pct'  : real_pct,
        'confidence': conf,
        'best_model': metadata['best_model'],
        'nlp'       : nlp,
    })


@app.route('/api/comparison_data')
def comparison_data():
    if not metadata:
        return jsonify({'error': 'No metadata'}), 500
    return jsonify(metadata.get('results', {}))


@app.route('/api/model_info')
def model_info():
    if not metadata:
        return jsonify({'ready': False})
    r    = metadata.get('results', {})
    best = metadata.get('best_model', '')
    return jsonify({
        'ready'      : best_model is not None,
        'best_model' : best,
        'models'     : list(r.keys()),
        'best_acc'   : round(r.get(best, {}).get('accuracy', 0) * 100, 2),
        'best_f1'    : round(r.get(best, {}).get('f1', 0) * 100, 2),
        'best_auc'   : round(r.get(best, {}).get('auc', 0) * 100, 2),
    })


if __name__ == '__main__':
    print("\n🛡  FakeShield running  →  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)