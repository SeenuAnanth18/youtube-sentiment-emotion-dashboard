import os
import gdown
import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing.text_preprocessing import clean_text
from models.sentiment_models import (
    textblob_sentiment,
    vader_sentiment,
    bert_sentiment,
    roberta_sentiment
)
from utils.ensemble import ensemble_voting


# ================= DOWNLOAD MODEL FROM GOOGLE DRIVE =================
MODEL_PATH = "sentiment_emotion_xlm_roberta.pth"
FILE_ID = "1OuI7uEYJVYwxlbB_Hs4nqt6EZHpG75iJ"


@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


download_model()


# ================= LAZY LOAD EMOTION MODEL =================
@st.cache_resource
def get_emotion_model():
    from models.emotion_model import detect_emotion
    return detect_emotion


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="YouTube Sentiment & Emotion Dashboard",
    page_icon="🎥",
    layout="wide"
)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color: #F4F6FA; }
.header {
    background: linear-gradient(90deg, #4b6cb7, #182848);
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.glass {
    background: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}
.footer {
    text-align: center;
    color: #7f8c8d;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h1>🎥 YouTube Sentiment & Emotion Analyzer</h1>
    <p>AI-powered NLP Dashboard for Multilingual Comments</p>
</div>
""", unsafe_allow_html=True)

# ================= INPUT SECTION =================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📝 Enter YouTube Comment")
    comment = st.text_area(
        "",
        height=120,
        placeholder="Example: This is very worst"
    )
    analyze = st.button("🚀 Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= ANALYSIS =================
if analyze and comment.strip():
    with st.spinner("Analyzing..."):
        clean_comment = clean_text(comment)

        model_results = [
            textblob_sentiment(clean_comment),
            vader_sentiment(clean_comment),
            bert_sentiment(clean_comment),
            roberta_sentiment(clean_comment)
        ]

        final_sentiment = ensemble_voting(model_results)

        # lazy load emotion model
        detect_emotion = get_emotion_model()
        emotions = detect_emotion(clean_comment)

        # Save history
        st.session_state.history.append({
            "Comment": comment,
            "Sentiment": final_sentiment,
            "Emotion": ", ".join(emotions)
        })

    # ================= KPI CARDS =================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Sentiment", final_sentiment)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dominant Emotion", emotions[0] if emotions else "Neutral")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= MODEL TABLE =================
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Model Predictions")
    st.table(pd.DataFrame({
        "Model": ["TextBlob", "VADER", "BERT", "RoBERTa"],
        "Prediction": model_results
    }))
    st.markdown('</div>', unsafe_allow_html=True)

    # ================= EMOTION TAGS =================
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("😊 Detected Emotions")
    st.write(" | ".join(emotions))
    st.markdown('</div>', unsafe_allow_html=True)

# ================= DASHBOARD =================
if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    st.markdown("## 📈 Live Analytics Dashboard")

    colA, colB = st.columns(2)

    with colA:
        fig_sent = px.histogram(
            df,
            x="Sentiment",
            color="Sentiment",
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with colB:
        emotion_counts = df["Emotion"].str.split(", ").explode().value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        fig_emo = px.pie(
            emotion_counts,
            names="Emotion",
            values="Count",
            title="Emotion Distribution"
        )
        st.plotly_chart(fig_emo, use_container_width=True)

    # ================= HISTORY TABLE =================
    st.markdown("## 🗂 Comment Analysis History")
    st.dataframe(df, use_container_width=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    AI-Based YouTube Sentiment & Emotion Analysis<br>
    NLP • Deep Learning • Streamlit Dashboard
</div>
""", unsafe_allow_html=True)git add app.p