# 📌 AI-Based Sentiment and Emotion Analysis of YouTube Comments

## 📖 Overview
This project is an AI-based system that analyzes YouTube comments to identify sentiment (positive, negative, neutral) and emotion (joy, anger, sadness, etc.).
It is mainly designed for tech product review comments, helping users understand opinions more clearly.

The system supports multiple languages such as Tamil, English, and Tanglish, and presents results using visual charts for easy understanding.

## 🚀 Features
- Analyze YouTube comments for sentiment and emotion
- Supports Tamil, English, and Tanglish
- Uses multiple models: TextBlob, VADER, BERT, RoBERTa
- Ensemble voting for accurate sentiment prediction
- Emotion detection using RoBERTa
- Dashboard view with history of analyzed comments
- Visualization using pie charts and bar charts
- Model-wise comparison output

## 🧠 Technologies Used
- Programming Language: Python
- Libraries: Pandas, NumPy, NLTK, Transformers
- Models: TextBlob, VADER, BERT, RoBERTa
- Framework: PyTorch / TensorFlow
- Visualization: Streamlit
- API: YouTube Data API v3

## ⚙️ System Workflow
1. User pastes a YouTube comment
2. Text preprocessing (cleaning and normalization)
3. Sentiment analysis using multiple models
4. Ensemble voting for final sentiment
5. Emotion detection using RoBERTa
6. Results displayed using charts
7. Dashboard updates with history of comments

## 📊 Output
- Cleaned comment
- Model-wise sentiment results
- Final sentiment
- Detected emotion
- Visualization (pie chart & bar chart)

## 💡 Use Case
- Analyze tech product reviews
- Understand user opinions quickly
- Compare sentiments across multiple comments
- Useful for students, researchers, and product analysis

## 🛠️ Installation

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py

## 📌 Requirements
- Python 3.8+
- Internet connection

## 📈 Future Enhancements
- Bulk comment analysis
- Real-time YouTube comment fetching
- More emotion categories
- Improved multilingual support

## 👨‍💻 Author
Seenu Ananth K

## ⭐ Conclusion
This project converts YouTube comments into meaningful insights using multiple AI models and visualization techniques.
