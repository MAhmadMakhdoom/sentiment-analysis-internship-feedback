
# ============================================================
# 🎯 Sentiment Analysis — Training Script
# Author   : MAhmadMakhdoom
# Dataset  : Capgemini Employee Reviews (AmbitionBox)
# Model    : Logistic Regression + TF-IDF
# ============================================================

# ── STEP 1 : IMPORT LIBRARIES ────────────────────────────────
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

nltk.download("stopwords")
print("✅ Libraries Loaded!")

# ── STEP 2 : LOAD DATASET ────────────────────────────────────
import kagglehub
import os

path  = kagglehub.dataset_download("manishkr1754/capgemini-employee-reviews-dataset")
files = os.listdir(path)
df    = pd.read_csv(os.path.join(path, files[0]))

print(f"✅ Dataset Loaded → {df.shape[0]} rows")

# ── STEP 3 : CREATE FEATURES & LABELS ────────────────────────
# Combine Likes & Dislikes into one feedback column
df["feedback_text"] = df["Likes"] + " " + df["Dislikes"]

# Convert Overall_rating to sentiment label
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "nutral"
    else:
        return "negative"

df["sentiment"] = df["Overall_rating"].apply(get_sentiment)
print("✅ Sentiment Labels Created!")
print(df["sentiment"].value_counts())

# ── STEP 4 : BALANCE DATASET ─────────────────────────────────
positive = df[df["sentiment"] == "positive"]
negative = df[df["sentiment"] == "negative"]
neutral  = df[df["sentiment"] == "nutral"]

# Undersample positive to match negative count
positive_downsampled = resample(
    positive,
    replace=False,
    n_samples=len(negative),
    random_state=42
)

df_balanced = pd.concat([positive_downsampled, negative, neutral])
df_balanced = df_balanced.copy()

print("✅ Dataset Balanced!")
print(df_balanced["sentiment"].value_counts())

# ── STEP 5 : CLEAN TEXT ───────────────────────────────────────
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()

df_balanced = df_balanced.dropna(subset=["feedback_text"])
df_balanced["clean_text"] = df_balanced["feedback_text"].apply(clean_text)

print("✅ Text Cleaned!")

# ── STEP 6 : TF-IDF VECTORIZATION ────────────────────────────
tfidf = TfidfVectorizer(max_features=5000)
X     = tfidf.fit_transform(df_balanced["clean_text"])
y     = df_balanced["sentiment"]

print(f"✅ TF-IDF Done → {X.shape[1]} features")

# ── STEP 7 : TRAIN TEST SPLIT ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Split Done → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── STEP 8 : TRAIN MODEL ──────────────────────────────────────
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

print("✅ Model Trained!")

# ── STEP 9 : EVALUATE ─────────────────────────────────────────
y_pred = model_lr.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ── STEP 10 : SAVE MODEL & VECTORIZER ─────────────────────────
with open("model_lr.pkl", "wb") as f:
    pickle.dump(model_lr, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model & TF-IDF Saved!")
print("🎯 Run app.py to launch Gradio UI!")
