# 🎯 Sentiment Analysis of Internship Feedback

Machine learning project to analyze intern feedback and detect sentiment using Logistic Regression and TF-IDF.

---

## 📋 Project Overview
| Item | Detail |
|------|--------|
| Dataset | Capgemini Employee Reviews (AmbitionBox) |
| Rows | 26,993 reviews |
| Model | Logistic Regression + TF-IDF |
| Classes | Positive / Negative / Neutral |
| Accuracy | 65% ✅ |

---

## 🗂️ Project Structure
```
├── train.py           → Data loading, cleaning, training
├── app.py             → Gradio UI for predictions
└── requirements.txt   → Dependencies
```

---

## 🔄 Project Journey

### Step 1 — Dataset
- Used Capgemini Employee Reviews from AmbitionBox
- Combined Likes & Dislikes as feedback text
- Converted Overall Rating to sentiment labels:
  - 4-5 → Positive ✅
  - 3   → Neutral 😐
  - 1-2 → Negative ❌

### Step 2 — Data Cleaning
- ✅ Removed stopwords
- ✅ Lowercased text
- ✅ Removed punctuation & special characters
- ✅ Balanced dataset via undersampling

### Step 3 — Model Results
| Class | Precision | Recall |
|-------|-----------|--------|
| Negative | 0.71 | 0.60 |
| Neutral  | 0.58 | 0.56 |
| Positive | 0.65 | 0.75 |
| **Accuracy** | | **65%** |

---

## 🚀 How To Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train model first
python train.py

# Launch Gradio app
python app.py
```

---

## 🛠️ Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Gradio
- Matplotlib
- WordCloud

---

## 👤 Author
**MAhmadMakhdoom**
> Future Data Scientist | ML & AI Enthusiast
