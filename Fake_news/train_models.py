"""
=============================================================
  FAKE NEWS DETECTOR — Fixed Training Script
  Properly handles Kaggle Bisaillon dataset:
    data/Fake.csv + data/True.csv
=============================================================
"""

import os, re, json, joblib, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
CLICKBAIT_KWS = [
    'breaking','secret','exposed','shocking','leaked','urgent','banned',
    'miracle','anonymous','bombshell','must read','wake up','sheeple',
    'elites','censored','mainstream','they dont want','deleted',
    'hidden truth','100% confirmed','you wont believe',
    'deep state','hoax','conspiracy','share before','going viral'
]

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
vader      = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def remove_agency_prefix(text):
    """
    Remove news agency location prefixes like:
    'WASHINGTON (Reuters) -', 'NEW YORK (AP) -'
    These are shortcuts the model memorizes instead of
    learning real language patterns.
    """
    text = re.sub(r'^[A-Z\s,]+\([^)]+\)\s*[-–—]*\s*', '', str(text))
    text = re.sub(r'^[A-Z\s,]+ - ', '', text)
    return text.strip()

def load_dataset():
    fake_path = os.path.join('data', 'Fake.csv')
    true_path = os.path.join('data', 'True.csv')

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("   ✓ Kaggle dataset found (Fake.csv + True.csv)")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        print(f"   Fake.csv columns: {fake_df.columns.tolist()}")
        print(f"   True.csv columns: {true_df.columns.tolist()}")

        # Use only title for training — titles are shorter, more generalizable
        # and don't contain Reuters/AP prefixes
        if 'title' in fake_df.columns:
            fake_texts = fake_df['title'].fillna('').astype(str)
            true_texts = true_df['title'].fillna('').astype(str)
            print("   Using: title column only (avoids agency prefix bias)")
        else:
            # fallback to text column but strip prefixes
            fake_texts = fake_df.iloc[:,0].fillna('').astype(str).apply(remove_agency_prefix)
            true_texts = true_df.iloc[:,0].fillna('').astype(str).apply(remove_agency_prefix)

        # Filter out very short texts
        fake_texts = fake_texts[fake_texts.str.split().str.len() >= 5]
        true_texts = true_texts[true_texts.str.split().str.len() >= 5]

        # Balance classes exactly
        n = min(len(fake_texts), len(true_texts), 10000)
        fake_texts = fake_texts.sample(n, random_state=42)
        true_texts = true_texts.sample(n, random_state=42)

        df = pd.DataFrame({
            'text' : pd.concat([fake_texts, true_texts], ignore_index=True),
            'label': [1]*n + [0]*n
        }).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"   Loaded: {len(df)} samples | Real: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")

        # Show sample to verify
        print("\n   Sample REAL title:", true_texts.iloc[0][:80])
        print("   Sample FAKE title:", fake_texts.iloc[0][:80])
        return df

    else:
        print("   ✗ data/Fake.csv or data/True.csv not found!")
        print("   Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        raise FileNotFoundError("Please add Fake.csv and True.csv to the data/ folder.")


# ─────────────────────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove news agency prefixes
    text = re.sub(r'^[a-z\s,]+\([^)]+\)\s*[-–—]*\s*', '', text)
    # Keep only letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords and stem
    words = [stemmer.stem(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)


# ─────────────────────────────────────────────────────────────
#  NLP FEATURES
# ─────────────────────────────────────────────────────────────
def extract_nlp_features(text):
    t = str(text)
    words = t.split()
    try:
        sentences = sent_tokenize(t)
    except Exception:
        sentences = [t]

    excl         = t.count('!') / (len(t) + 1)
    caps         = sum(1 for w in words if w.isupper() and len(w) > 2) / (len(words) + 1)
    qmarks       = t.count('?') / (len(sentences) + 1)
    cb_score     = sum(1 for kw in CLICKBAIT_KWS if kw in t.lower()) / len(CLICKBAIT_KWS)
    vs           = vader.polarity_scores(t)
    sent_ext     = abs(vs['compound'])
    neg_ratio    = vs['neg']
    blob         = TextBlob(t)
    subjectivity = blob.sentiment.subjectivity
    pol_ext      = abs(blob.sentiment.polarity)
    lex_div      = len(set(words)) / (len(words) + 1)
    avg_wl       = np.mean([len(w) for w in words]) if words else 0
    sent_lens    = [len(s.split()) for s in sentences]
    sent_var     = float(np.var(sent_lens)) if len(sent_lens) > 1 else 0
    punc_abuse   = (t.count('!') + t.count('?')) / (len(t) / 100 + 1)
    has_numbers  = 1 if re.search(r'\d+', t) else 0
    has_quotes   = 1 if ('"' in t or "'" in t) else 0
    text_len     = float(np.log1p(len(t)))

    return [excl, caps, qmarks, cb_score, sent_ext, neg_ratio,
            subjectivity, pol_ext, lex_div, avg_wl, sent_var,
            punc_abuse, has_numbers, has_quotes, text_len]


# ─────────────────────────────────────────────────────────────
#  CUSTOM SKLEARN CLASSES
# ─────────────────────────────────────────────────────────────
class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):   return X


class NLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
    def fit(self, X, y=None):
        feats = np.array([extract_nlp_features(t) for t in X], dtype=np.float32)
        self.scaler.fit(feats)
        return self
    def transform(self, X):
        feats = np.array([extract_nlp_features(t) for t in X], dtype=np.float32)
        return self.scaler.transform(feats)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_params=None):
        self.tfidf_params = tfidf_params or {}
        self.tfidf        = TfidfVectorizer(**self.tfidf_params)
        self.nlp_scaler   = MinMaxScaler()

    def fit(self, X, y=None):
        cleaned = [clean_text(t) for t in X]
        self.tfidf.fit(cleaned)
        nlp_feats = np.array([extract_nlp_features(t) for t in X], dtype=np.float32)
        self.nlp_scaler.fit(nlp_feats)
        return self

    def transform(self, X):
        cleaned    = [clean_text(t) for t in X]
        tfidf_mat  = self.tfidf.transform(cleaned)
        nlp_feats  = np.array([extract_nlp_features(t) for t in X], dtype=np.float32)
        nlp_scaled = self.nlp_scaler.transform(nlp_feats)
        return sp.hstack([tfidf_mat, sp.csr_matrix(nlp_scaled)])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────
def train_and_evaluate():
    print("\n" + "="*62)
    print("   FAKE NEWS DETECTOR — TRAINING")
    print("="*62)

    # 1. Load data
    print("\n[1/6] Loading dataset …")
    df = load_dataset()
    X, y = df['text'].values, df['label'].values

    # 3-way split: 70% train / 15% val / 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    print(f"\n   Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    # 2. Features
    print("\n[2/6] Building TF-IDF + NLP features …")
    # Conservative settings — generalize better
    TFIDF = dict(
        max_features = 10000,
        ngram_range  = (1, 2),
        min_df       = 3,
        max_df       = 0.85,
        sublinear_tf = True,
    )
    TFIDF_NB = dict(
        max_features = 10000,
        ngram_range  = (1, 2),
        min_df       = 3,
        max_df       = 0.85,
    )

    feat_lr  = CombinedFeatures(TFIDF)
    feat_rf  = CombinedFeatures(TFIDF)
    feat_nb  = CombinedFeatures(TFIDF_NB)
    feat_xgb = CombinedFeatures(TFIDF)

    print("   Fitting Logistic Regression features …")
    Xtr_lr  = feat_lr.fit_transform(X_train);  Xte_lr  = feat_lr.transform(X_test)
    print("   Fitting Random Forest features …")
    Xtr_rf  = feat_rf.fit_transform(X_train);  Xte_rf  = feat_rf.transform(X_test)
    print("   Fitting Naive Bayes features …")
    Xtr_nb  = feat_nb.fit_transform(X_train);  Xte_nb  = feat_nb.transform(X_test)
    print("   Fitting XGBoost features …")
    Xtr_xgb = feat_xgb.fit_transform(X_train); Xte_xgb = feat_xgb.transform(X_test)

    Xtr_nb_pos = sp.csr_matrix(np.abs(Xtr_nb.toarray()))
    Xte_nb_pos = sp.csr_matrix(np.abs(Xte_nb.toarray()))

    # 3. Models
    print("\n[3/6] Training 4 models …")
    configs = {
        'Logistic Regression': (
            LogisticRegression(
                C             = 1.0,
                max_iter      = 1000,
                solver        = 'lbfgs',
                class_weight  = 'balanced',
                random_state  = 42
            ),
            Xtr_lr, Xte_lr, feat_lr, False
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators  = 200,
                max_depth     = 15,
                min_samples_leaf = 4,
                max_features  = 'sqrt',
                class_weight  = 'balanced',
                random_state  = 42,
                n_jobs        = -1
            ),
            Xtr_rf, Xte_rf, feat_rf, False
        ),
        'Naive Bayes': (
            MultinomialNB(alpha=0.5),
            Xtr_nb_pos, Xte_nb_pos, feat_nb, True
        ),
        'XGBoost': (
            xgb.XGBClassifier(
                n_estimators      = 300,
                max_depth         = 5,
                learning_rate     = 0.05,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                reg_alpha         = 0.5,
                reg_lambda        = 1.0,
                eval_metric       = 'logloss',
                random_state      = 42,
                n_jobs            = -1
            ),
            Xtr_xgb, Xte_xgb, feat_xgb, False
        ),
    }

    results = {}
    trained = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (clf, Xtr, Xte, feat, is_nb) in configs.items():
        print(f"\n   ▸ Training {name} …")
        clf.fit(Xtr, y_train)

        # Find best threshold on validation set
        Xval = feat.transform(X_val)
        if is_nb:
            Xval = sp.csr_matrix(np.abs(Xval.toarray()))
        val_probs = clf.predict_proba(Xval)[:, 1]

        best_thresh, best_f1v = 0.5, 0.0
        for t in np.arange(0.25, 0.76, 0.05):
            preds = (val_probs >= t).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1v:
                best_f1v   = score
                best_thresh = t
        print(f"     Tuned threshold: {best_thresh:.2f}  (val F1: {best_f1v:.4f})")

        # Evaluate on test set
        test_probs = clf.predict_proba(Xte)[:, 1]
        y_pred     = (test_probs >= best_thresh).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, test_probs)

        cv_scores = cross_val_score(clf, Xtr, y_train,
                                    cv=cv, scoring='f1', n_jobs=-1)

        print(f"     Acc:{acc:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1:.4f}  AUC:{auc:.4f}")
        print(f"     CV  F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        gap = abs(f1 - cv_scores.mean())
        status = "⚠ Overfitting" if gap > 0.08 else "✓ OK"
        print(f"     Gap: {gap:.4f}  {status}")

        results[name] = {
            'accuracy'        : float(acc),
            'precision'       : float(prec),
            'recall'          : float(rec),
            'f1'              : float(f1),
            'auc'             : float(auc),
            'cv_f1_mean'      : float(cv_scores.mean()),
            'cv_f1_std'       : float(cv_scores.std()),
            'threshold'       : float(best_thresh),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        trained[name] = {
            'clf'      : clf,
            'feat'     : feat,
            'nb'       : is_nb,
            'threshold': float(best_thresh),
        }

    # 4. Charts
    print("\n[4/6] Generating charts …")
    os.makedirs('frontend/static/img', exist_ok=True)
    _chart_comparison(results)
    _chart_confusion(results)
    _chart_radar(results)
    print("   ✓ Charts saved")

    # 5. Pick best model by CV F1
    best_name = max(results, key=lambda x: results[x]['cv_f1_mean'])

    # 6. Save
    print(f"\n[5/6] Best model: {best_name}")
    print("[6/6] Saving …")
    os.makedirs('models', exist_ok=True)
    joblib.dump(trained[best_name], 'models/best_model.pkl')
    joblib.dump(trained,            'models/all_models.pkl')
    with open('models/metadata.json', 'w') as f:
        json.dump({'best_model': best_name, 'results': results}, f, indent=2)

    # Final summary
    print("\n" + "="*62)
    print("  ✅ TRAINING COMPLETE")
    print("="*62)
    print(f"\n  {'Model':<22} {'Acc':>7} {'F1':>7} {'CV-F1':>7} {'AUC':>7} {'Thr':>5}")
    print("  " + "-"*55)
    for n, r in sorted(results.items(), key=lambda x: -x[1]['cv_f1_mean']):
        star = " ← BEST" if n == best_name else ""
        print(f"  {n:<22} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
              f"{r['cv_f1_mean']:>7.4f} {r['auc']:>7.4f} "
              f"{r['threshold']:>5.2f}{star}")
    print("\n  Run:  python backend/app.py")
    print("  Open: http://127.0.0.1:5000\n")


# ─────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────
DARK='#0d1117'; SURF='#161b22'; BORDER='#21262d'
CYAN='#00e5ff'; GREEN='#00ff88'; RED='#ff3d5a'; YELL='#ffd600'
COLORS=[CYAN, GREEN, RED, YELL]

def _ax(ax):
    ax.set_facecolor(SURF); ax.spines[:].set_color(BORDER)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.grid(alpha=0.15, color='#8b949e')

def _chart_comparison(results):
    names  = list(results.keys())
    short  = ['LR','RF','NB','XGB']
    mets   = ['accuracy','precision','recall','f1','auc']
    mlbls  = ['Accuracy','Precision','Recall','F1','AUC']
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.patch.set_facecolor(DARK)
    ax=axes[0]; _ax(ax)
    x=np.arange(len(names)); w=0.15
    for i,(m,l) in enumerate(zip(mets,mlbls)):
        ax.bar(x+i*w,[results[n][m] for n in names],w,label=l,color=COLORS[i%4],alpha=0.85)
    ax.set_xticks(x+w*2); ax.set_xticklabels(short,color='#c9d1d9',fontsize=9)
    ax.set_ylim(0,1.1); ax.set_ylabel('Score',color='#c9d1d9',fontsize=9)
    ax.set_title('All Metrics',color='white',fontsize=12,fontweight='bold')
    ax.legend(facecolor=SURF,edgecolor=BORDER,labelcolor='#c9d1d9',fontsize=8)
    ax2=axes[1]; _ax(ax2)
    x2=np.arange(len(names))
    ax2.bar(x2-0.2,[results[n]['f1'] for n in names],0.35,label='Test F1',color=CYAN,alpha=0.85)
    ax2.bar(x2+0.2,[results[n]['cv_f1_mean'] for n in names],0.35,label='CV F1',
            color=GREEN,alpha=0.85,
            yerr=[results[n]['cv_f1_std'] for n in names],
            capsize=4,error_kw={'color':'white','linewidth':1.2})
    ax2.set_xticks(x2); ax2.set_xticklabels(short,color='#c9d1d9',fontsize=9)
    ax2.set_ylim(0,1.1); ax2.set_ylabel('F1',color='#c9d1d9',fontsize=9)
    ax2.set_title('Test F1 vs CV F1',color='white',fontsize=12,fontweight='bold')
    ax2.legend(facecolor=SURF,edgecolor=BORDER,labelcolor='#c9d1d9',fontsize=8)
    ax2.tick_params(colors='#c9d1d9')
    plt.tight_layout(pad=2)
    plt.savefig('frontend/static/img/comparison.png',dpi=130,bbox_inches='tight',facecolor=DARK)
    plt.close()

def _chart_confusion(results):
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Confusion Matrices',color='white',fontsize=13,fontweight='bold')
    for ax,(name,res) in zip(axes,results.items()):
        ax.set_facecolor(SURF)
        sns.heatmap(np.array(res['confusion_matrix']),annot=True,fmt='d',ax=ax,
                    cmap='Blues',xticklabels=['Real','Fake'],
                    yticklabels=['Real','Fake'],linewidths=0.5,cbar=False)
        ax.set_title(name,color='white',fontsize=9,fontweight='bold')
        ax.tick_params(colors='#c9d1d9',labelsize=8)
        ax.set_xlabel('Predicted',color='#8b949e',fontsize=8)
        ax.set_ylabel('Actual',color='#8b949e',fontsize=8)
    plt.tight_layout()
    plt.savefig('frontend/static/img/confusion.png',dpi=130,bbox_inches='tight',facecolor=DARK)
    plt.close()

def _chart_radar(results):
    cats=['Accuracy','Precision','Recall','F1','AUC']
    N=len(cats)
    angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(SURF)
    ax.spines['polar'].set_color(BORDER)
    ax.tick_params(colors='#8b949e',labelsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats,color='#c9d1d9',fontsize=9)
    ax.set_ylim(0,1)
    for i,(name,sname) in enumerate(zip(results.keys(),['LR','RF','NB','XGB'])):
        r=results[name]
        vals=[r['accuracy'],r['precision'],r['recall'],r['f1'],r['auc']]+[r['accuracy']]
        ax.plot(angles,vals,color=COLORS[i],linewidth=2,label=sname)
        ax.fill(angles,vals,color=COLORS[i],alpha=0.07)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),
              facecolor=SURF,edgecolor=BORDER,labelcolor='#c9d1d9',fontsize=9)
    ax.set_title('Model Radar',color='white',fontsize=12,fontweight='bold',pad=20)
    plt.tight_layout()
    plt.savefig('frontend/static/img/radar.png',dpi=130,bbox_inches='tight',facecolor=DARK)
    plt.close()


if __name__ == '__main__':
    train_and_evaluate()