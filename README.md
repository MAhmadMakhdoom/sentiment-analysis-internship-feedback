# 🎯 Sentiment Analysis of Internship Feedback

> *"Understanding how interns feel — one review at a time."*

---

## 📌 Project Overview

This project analyzes real employee and intern feedback to automatically detect sentiment — Positive, Negative, or Neutral — using Machine Learning. Built as part of an internship program to help organizations understand intern satisfaction and identify areas for improvement.

| Item | Detail |
|------|--------|
| Dataset | Capgemini Employee Reviews (AmbitionBox) |
| Total Reviews | 26,993 |
| Model | Logistic Regression + TF-IDF |
| Classes | Positive / Negative / Neutral |
| Final Accuracy | 65% ✅ |

---

## 🗂️ Project Structure
```
├── train.py           → Full training pipeline
├── app.py             → Gradio UI for predictions
├── requirements.txt   → All dependencies
└── README.md          → Project documentation
```

---

## 🔄 Complete Project Journey

### Step 1 — Dataset Discovery & Selection
Finding the right dataset was the first and hardest challenge of this project. Intern-specific datasets are extremely rare publicly. After exploring multiple sources including Kaggle, UCI Repository and Hugging Face, I discovered the Capgemini Employee Reviews dataset from AmbitionBox. The key insight was that employee reviews closely mirror intern feedback in structure and content — making it a perfect fit for this task.

### Step 2 — Feature Engineering
The dataset had 14 columns but most were numerical ratings. The real challenge was identifying which columns carried actual human sentiment. I combined the Likes and Dislikes text columns into a single feedback_text feature and converted the Overall Rating into sentiment labels:
- Rating 4-5 → Positive
- Rating 3 → Neutral
- Rating 1-2 → Negative

### Step 3 — Handling Class Imbalance
After creating labels, I discovered a significant imbalance:
- Positive → 16,834 reviews
- Negative → 5,454 reviews
- Neutral → 4,705 reviews

A model trained on this imbalanced data would simply predict Positive for everything and still appear accurate — a classic ML trap. I applied undersampling to reduce the Positive class and balance all three classes equally, ensuring the model learns fairly from all sentiments.

### Step 4 — Text Cleaning
Raw text data is never clean. Real reviews contained uppercase letters, punctuation, special characters, emojis, stopwords and line breaks — all of which add noise without adding meaning. I built a custom cleaning pipeline:
- Lowercased all text
- Removed punctuation and special characters
- Removed stopwords (the, is, a, etc.)
- Stripped extra whitespace

### Step 5 — TF-IDF Vectorization
Machines cannot understand text directly. TF-IDF converts text into numbers by giving high scores to rare meaningful words and low scores to common words. This was the bridge between raw human language and machine understanding.

### Step 6 — Model Training & Evaluation
Logistic Regression with TF-IDF achieved 65% accuracy — right at the industry benchmark for sentiment analysis on real world text data.

---

## 📊 Final Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.71 | 0.60 | 0.65 |
| Neutral | 0.58 | 0.56 | 0.57 |
| Positive | 0.65 | 0.75 | 0.70 |
| **Accuracy** | | | **65%** |

---

## 💡 Key Challenges & Learnings

### Challenge 1 — Dataset Selection
There is no publicly available intern-specific feedback dataset. The solution was to think critically and find the closest alternative — employee reviews — and adapt it for the purpose. This taught me that in real ML projects, perfect data rarely exists. You work with what is available and make it suitable.

### Challenge 2 — Class Imbalance
Discovering that 62% of data was Positive was a critical moment. Without fixing this, the model would have been biased and unreliable. Applying undersampling was not just a technical fix — it was understanding the deeper problem of fairness in machine learning.

### Challenge 3 — Text is Messy
Unlike numerical data, text data is unpredictable. People write in different styles, use slang, make typos and mix languages. Building a robust cleaning pipeline that handles all these cases without losing meaning was a real challenge.

---

## ✅ Best Practices Applied

- Used real world dataset instead of fully synthetic data
- Applied class balancing before training
- Built reusable clean_text() function
- Separated training code (train.py) from UI code (app.py)
- Used TF-IDF max_features to control dimensionality
- Evaluated per-class performance not just overall accuracy

---

## 🚀 How To Run
```bash
# Step 1 → Install dependencies
pip install -r requirements.txt

# Step 2 → Train model
python train.py

# Step 3 → Launch app
python app.py
```

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Gradio
- Matplotlib & WordCloud

---

## 👤 Author

**MAhmadMakhdoom**
> Future Data Scientist | ML & AI Enthusiast
> Building models today. Solving real problems tomorrow.
