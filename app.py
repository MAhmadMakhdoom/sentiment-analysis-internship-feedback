
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# Load model & vectorizer
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()

history = []

def predict_sentiment(text):
    clean      = clean_text(text)
    vectorized = tfidf.transform([clean])
    prediction    = model_lr.predict(vectorized)[0]
    probabilities = model_lr.predict_proba(vectorized)[0]
    classes       = model_lr.classes_
    confidence    = dict(zip(classes, (probabilities * 100).round(2)))

    if prediction == "positive":
        label = "✅ Positive — Intern is Happy!"
    elif prediction == "negative":
        label = "❌ Negative — Intern is Unhappy!"
    else:
        label = "😐 Neutral — Intern has Mixed Feelings!"

    conf_text = (
        f"🔵 Negative : {confidence.get('negative', 0)}%\n"
        f"⚪ Neutral  : {confidence.get('nutral',   0)}%\n"
        f"🟢 Positive : {confidence.get('positive', 0)}%"
    )

    values = [
        confidence.get("negative", 0),
        confidence.get("nutral",   0),
        confidence.get("positive", 0)
    ]

    fig_meter, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(["Sentiment"], [values[0]], color="#e74c3c", label="Negative")
    ax.barh(["Sentiment"], [values[1]], left=[values[0]], color="#95a5a6", label="Neutral")
    ax.barh(["Sentiment"], [values[2]], left=[values[0]+values[1]], color="#2ecc71", label="Positive")
    ax.set_xlim(0, 100)
    ax.set_title("Sentiment Meter")
    ax.legend(loc="upper right")
    plt.tight_layout()

    fig_wc, ax2 = plt.subplots(figsize=(6, 3))
    if clean.strip():
        wc = WordCloud(width=600, height=300, background_color="white", colormap="RdYlGn").generate(clean)
        ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title("Keywords Word Cloud")
    plt.tight_layout()

    history.append({
        "Feedback"  : text[:50] + "..." if len(text) > 50 else text,
        "Sentiment" : prediction.capitalize(),
        "Confidence": f"{max(values):.1f}%"
    })

    return label, conf_text, fig_meter, fig_wc, pd.DataFrame(history)

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(title="Intern Feedback Analyzer", theme=theme) as app:
    gr.Markdown("# 🎯 Intern Feedback Sentiment Analyzer")
    gr.Markdown("Analyze intern feedback to detect sentiment instantly.")
    with gr.Row():
        text_input = gr.Textbox(label="Enter Intern Feedback", placeholder="Type feedback here...", lines=3)
    analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
    with gr.Row():
        sentiment_out  = gr.Text(label="Sentiment Result")
        confidence_out = gr.Text(label="Confidence Scores")
    with gr.Row():
        meter_out = gr.Plot(label="Sentiment Meter")
        wc_out    = gr.Plot(label="Word Cloud")
    history_out = gr.Dataframe(label="📋 Feedback History")
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[sentiment_out, confidence_out, meter_out, wc_out, history_out]
    )

app.launch()
