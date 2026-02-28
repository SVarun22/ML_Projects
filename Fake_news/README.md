# 🛡️ FakeShield — AI Fake News Detector

Ever read a headline and thought *"wait, is this actually real?"* FakeShield helps you answer that question instantly. Paste any news article or headline and get back an exact percentage showing how fake or real it is — powered by four machine learning models working together behind the scenes.

---

## 🗂️ How the Project is Organized

Here's everything in the folder and what each piece does:

```
Fake_news/
│
├── train_models.py         ← Run this first. Trains all 4 models on real data.
├── requirements.txt        ← All the Python packages the project needs.
├── README.md               ← This file.
│
├── backend/
│   └── app.py              ← The Flask server. Run this to start the website.
│
├── frontend/
│   ├── templates/
│   │   ├── index.html      ← The main detector page you interact with.
│   │   └── compare.html    ← See how all 4 models performed side by side.
│   └── static/
│       └── img/            ← Charts get saved here automatically after training.
│
├── models/                 ← Gets created automatically when you train.
│   ├── best_model.pkl      ← The winning model, saved and ready to use.
│   ├── all_models.pkl      ← All 4 models saved together.
│   └── metadata.json       ← Scores, thresholds, and model info.
│
└── data/                   ← Put your dataset files here before training.
    ├── Fake.csv            ← ~23,000 fake news articles from Kaggle.
    └── True.csv            ← ~21,000 real news articles from Kaggle.
```

---

## 📦 Before You Start — Get the Dataset

The model needs real data to learn from. Download it free from Kaggle:

👉 **https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset**

You'll need a free Kaggle account. Once you download and unzip it, you'll get two files — `Fake.csv` and `True.csv`. Drop both of them into the `data/` folder inside your project. That's it.

---

## 🚀 Getting It Running

Open your terminal in PyCharm (`Alt + F12`) and follow these steps in order.

### First time setup

```powershell
# Go to your project folder
cd C:\Users\YourName\Desktop\Fake_news

# Create a virtual environment
python -m venv venv

# Activate it — you'll see (venv) appear on the left
venv\Scripts\activate

# Install everything the project needs
pip install flask flask-cors scikit-learn nltk joblib matplotlib seaborn textblob vaderSentiment scipy xgboost

# Download the language data NLTK needs
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# Train the models — grab a coffee, this takes about 5 minutes
python train_models.py

# Start the website
python backend/app.py
```

Now open your browser and go to **http://127.0.0.1:5000** 🎉

---

### Every day after that

Only 3 commands needed:

```powershell
cd C:\Users\YourName\Desktop\Fake_news
venv\Scripts\activate
python backend/app.py
```

> ⚠️ Don't close the terminal while using the app. Press `Ctrl+C` when you want to stop it.

---

## 🤖 The Four Models

FakeShield doesn't rely on just one model — it trains four different ones and automatically picks the best performer. Here's what each one brings to the table:

| Model | What makes it useful |
|-------|---------------------|
| **Logistic Regression** | Simple, fast, and surprisingly good at text. Gives well-calibrated probabilities. |
| **Random Forest** | Builds 200 decision trees and combines their votes. Great at catching non-obvious patterns. |
| **Naive Bayes** | A classic for text classification. Lightweight but effective, especially on short headlines. |
| **XGBoost** | The heavy hitter. Gradient boosted trees with regularization — usually ends up as the best model. |

After training, the app compares all four using cross-validation and automatically uses the best one for your predictions.

---

## 🧠 The 15 NLP Signals

Beyond just reading the words, FakeShield measures 15 specific signals in every piece of text. Think of them as red flags the model looks for:

| Signal | Why it matters |
|--------|---------------|
| **Clickbait Keywords** | Words like BREAKING, SECRET, EXPOSED, SHOCKING are way more common in fake news |
| **ALL-CAPS Ratio** | Shouting in text is a classic manipulation tactic |
| **Exclamation Marks** | Real journalism rarely ends sentences with !!! |
| **Emotional Extremity** | Fake news tends to be emotionally extreme — either outrage or miracle |
| **Subjectivity Score** | Fact-based writing scores low on subjectivity. Opinion-heavy text scores high. |
| **Negative Sentiment** | Fake news disproportionately uses fear and anger to keep you reading |
| **Punctuation Abuse** | Excessive ? and ! patterns are a giveaway |
| **Lexical Diversity** | Real articles use varied vocabulary. Fake ones repeat the same loaded words. |
| **Numeric Citations** | "Studies show" vs "A 2023 study of 12,400 patients showed" — numbers add credibility |
| **Quoted Sources** | Legitimate news attributes claims. Fake news rarely does. |
| **Polarity Extremity** | Everything is either amazing or catastrophic in fake news |
| **Average Word Length** | Academic and journalistic writing tends to use longer, more precise words |
| **Sentence Length Variance** | Rambling, inconsistent sentence structure is common in fabricated content |
| **Question Mark Density** | Rhetorical questions are a common manipulation device |
| **Text Length** | Very short or very long texts follow different patterns |

---

## 📊 What You See on the Website

**Main page** (`/`)

You paste your article, hit Analyze, and instantly get:
- Two animated ring gauges — one showing **Fake %** and one showing **Real %**
- A clear verdict banner saying FAKE or REAL with a confidence score
- A full breakdown of all 15 NLP signals, each with its own score
- Any suspicious keywords or ALL-CAPS words highlighted as chips

**Comparison page** (`/compare`)

After training, head here to see how the four models stacked up:
- A metrics table with Accuracy, Precision, Recall, F1 and AUC for each model
- A bar chart comparing all metrics side by side
- A Test F1 vs Cross-Validation F1 chart to check for overfitting
- Confusion matrices showing where each model gets it right and wrong
- A radar chart giving a visual overview of each model's strengths

---

## 🛠️ Built With

| Layer | Tools |
|-------|-------|
| Backend | Python 3.11, Flask |
| Machine Learning | scikit-learn, XGBoost |
| NLP | NLTK, TextBlob, VADER Sentiment |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Visualizations | Matplotlib, Seaborn |

---

## 🔧 Something Not Working?

| What you're seeing | What to do |
|--------------------|-----------|
| `ModuleNotFoundError` | You forgot to run `venv\Scripts\activate` |
| `Models not found` | Run `python train_models.py` first |
| `TemplateNotFound: index.html` | Make sure `index.html` is inside `frontend/templates/` not just `frontend/` |
| `FileNotFoundError: Fake.csv` | Put `Fake.csv` and `True.csv` inside the `data/` folder |
| Everything classified as Fake | Delete the `models/` folder and retrain — the old model was biased |
| Port 5000 already in use | Change `port=5000` to `port=5001` at the bottom of `backend/app.py` |
| NLTK errors | Run the NLTK download command from the setup section above |

---

## 💡 A Few Tips

- Hit `Ctrl + Enter` in the text box to analyze without clicking the button
- The **Real News** and **Fake News** sample buttons are great for testing
- You only ever need to run `train_models.py` once — the trained models are saved
- The model works best on English-language news headlines and short articles
- News from after 2018 may score slightly differently since the training data is from 2015–2018