"""
student.py – ML data loading, models, prediction and student-facing dashboard.
"""

import os
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from db import get_all_resources

DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA.xlsx")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource(show_spinner="Loading dataset…")
def load_data():
    """Reads DATA.xlsx, selects features, splits dataset."""
    df = pd.read_excel(DATA_PATH)

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean().round(0).astype(int))

    for col in ["GENDER", "EDUCATIONAL QUALIFICATION"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    label_encoders = {}
    for col in ["GENDER", "EDUCATIONAL QUALIFICATION", "LABEL"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    correlation = df.corr(numeric_only=True)["LABEL"].abs()
    selected_features = correlation[correlation > 0.2].index.tolist()
    if "LABEL" in selected_features:
        selected_features.remove("LABEL")
    if not selected_features:
        selected_features = [c for c in df.select_dtypes(include="number").columns if c != "LABEL"]

    X = df[selected_features]
    y = df["LABEL"]

    feature_means = {f: float(X[f].mean()) for f in selected_features}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, selected_features, label_encoders, feature_means

@st.cache_resource(show_spinner="Training models…")
def train_models(_X_train, _y_train, _X_test, _y_test):
    """Trains XGBoost, SVM, and AdaBoost classifiers."""
    models = {}

    xgb = XGBClassifier(eval_metric="mlogloss", n_estimators=50, learning_rate=0.2, max_depth=4, random_state=42)
    xgb.fit(_X_train, _y_train)
    models["XGBoost"] = (xgb, accuracy_score(_y_test, xgb.predict(_X_test)) * 100)

    svm = SVC(kernel="linear", C=1.0, random_state=42)
    svm.fit(_X_train, _y_train)
    models["SVM"] = (svm, accuracy_score(_y_test, svm.predict(_X_test)) * 100)

    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.1, random_state=42)
    ada.fit(_X_train, _y_train)
    models["AdaBoost"] = (ada, accuracy_score(_y_test, ada.predict(_X_test)) * 100)

    return models

def predict(model, input_df: pd.DataFrame, label_encoder: LabelEncoder):
    """Run inference on a single-row DataFrame."""
    prediction = model.predict(input_df)[0]
    return label_encoder.inverse_transform([int(prediction)])[0]

def _clean_feature_name(feature: str) -> str:
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', feature).strip()
    if cleaned == cleaned.upper():
        cleaned = cleaned.title()
    return cleaned

def detect_weak_topics(input_data: dict, feature_means: dict) -> list:
    """Identify features where the student scored below the dataset mean."""
    weak = []
    for feature, value in input_data.items():
        if feature in feature_means and isinstance(value, (int, float)):
            mean = feature_means[feature]
            if value < mean:
                weak.append((_clean_feature_name(feature), value, mean, mean - value))
    weak.sort(key=lambda x: x[3], reverse=True)
    return [(name, score, mean) for name, score, mean, _ in weak[:5]]

_BASE_SUGGESTIONS = {
    "poor": [
        "📘 Start with foundational courses on platforms like **Coursera** or **Khan Academy**.",
        "✍️ Practice daily — even 30 minutes of focused study builds strong habits.",
        "🔗 Join beginner-friendly communities (e.g., Reddit r/learnprogramming, Discord servers).",
        "📝 Keep a learning journal to track progress and revisit weak areas.",
        "🎯 Set small, achievable weekly goals to build confidence.",
    ],
    "average": [
        "🚀 Take on real-world projects — build a portfolio on **GitHub**.",
        "📖 Explore advanced topics through **MIT OpenCourseWare** or **Udemy**.",
        "🤝 Collaborate with peers via hackathons or open-source contributions.",
        "🔍 Practice problem-solving on platforms like **LeetCode** or **HackerRank**.",
        "📊 Analyse your weak skill areas and create a focused improvement plan.",
    ],
    "good": [
        "🏆 Contribute to cutting-edge open-source projects or publish research.",
        "🎓 Consider mentoring beginners — teaching reinforces expertise.",
        "📚 Stay current with journals, arXiv papers, and conference talks (NeurIPS, ICML).",
        "🛠️ Specialise in a niche (e.g., MLOps, NLP, CV) to differentiate yourself.",
        "💼 Build a strong professional network through LinkedIn and academic conferences.",
    ],
}
_DEFAULT_SUGGESTIONS = [
    "📌 Review your inputs carefully.",
    "📖 Dedicate time each week to studying the topics covered in the assessment.",
    "💬 Speak to your faculty for additional guidance.",
]

def get_suggestions(label: str) -> list:
    """Get the base targeted tips for the predicted category."""
    label_key = label.lower().strip()
    suggestions = list(_BASE_SUGGESTIONS.get(label_key, _DEFAULT_SUGGESTIONS))
    if label_key == "poor":
        suggestions.append("💡 Try the **Google Digital Garage** free certification to build foundational skills.")
    elif label_key == "average":
        suggestions.append("💡 Earn an industry certification (e.g., **AWS**, **Google Cloud**, **Azure**) to validate skills.")
    elif label_key == "good":
        suggestions.append("💡 Apply for competitive fellowships or scholarships to fund further research.")
    return suggestions

def _logout():
    """Clear session state and return to home."""
    for key in ["logged_in", "username", "full_name", "role", "page"]:
        st.session_state.pop(key, None)
    st.session_state.page = "home"
    st.rerun()

def show_student_dashboard():
    """Render the Student Dashboard."""
    if not st.session_state.get("logged_in") or st.session_state.get("role") != "student":
        st.error("🚫 Access denied. Please log in as a student.")
        st.stop()

    with st.sidebar:
        st.markdown(
            f"""
            <div class='sidebar-profile'>
                <div class='avatar'>👤</div>
                <h3>{st.session_state.get('full_name', 'Student')}</h3>
                <span class='role-badge student'>Student</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    st.markdown("<h2 class='page-title'>🎓 Student Dashboard</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔬 Skill Analysis", "📂 Faculty Resources"])

    with tab1:
        st.markdown("### Enter Your Skill Scores")
        st.caption("Fill in your scores for each skill area, then click **Predict** to see your level.")

        X_train, X_test, y_train, y_test, selected_features, label_encoders, feature_means = load_data()
        models = train_models(X_train, y_train, X_test, y_test)

        input_data = {}
        with st.form("prediction_form"):
            cols = st.columns(2)
            for idx, feature in enumerate(selected_features):
                with cols[idx % 2]:
                    if feature == "EDUCATIONAL QUALIFICATION":
                        edu_options = {"PhD": 0, "Postgraduate (PG)": 1, "Undergraduate (UG)": 2}
                        selected_edu = st.selectbox("Educational Qualification", list(edu_options.keys()))
                        input_data[feature] = edu_options[selected_edu]
                    elif feature == "GENDER":
                        gender_options = {"Male": 0, "Female": 1}
                        selected_g = st.selectbox("Gender", list(gender_options.keys()))
                        input_data[feature] = gender_options[selected_g]
                    else:
                        label = feature.title().replace("_", " ")
                        input_data[feature] = st.number_input(label, min_value=0, max_value=100, value=0, step=1)

            algo = st.selectbox("Choose Algorithm", list(models.keys()), help="All algorithms have been trained on your institution's dataset.")
            predict_btn = st.form_submit_button("🔍 Predict My Skill Level", use_container_width=True)

        if predict_btn:
            model, accuracy = models[algo]
            input_df = pd.DataFrame([input_data])[X_train.columns]
            label = predict(model, input_df, label_encoders["LABEL"])

            badge_colors = {"poor": "#e74c3c", "average": "#f39c12", "good": "#27ae60"}
            color = badge_colors.get(label.lower(), "#3498db")

            st.markdown(
                f"""
                <div class='result-card'>
                    <h4>Prediction Result</h4>
                    <div class='prediction-badge' style='background:{color};'>{label}</div>
                    <p>Algorithm: <b>{algo}</b> &nbsp;|&nbsp; Accuracy: <b>{accuracy:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("### 💡 Personalised Suggestions")

            for tip in get_suggestions(label):
                st.markdown(f"- {tip}")

            weak_topics = detect_weak_topics(input_data, feature_means)
            if weak_topics:
                st.markdown("#### 🎯 Focus Areas to Improve")
                st.markdown("We noticed you scored below average in these specific areas. Click the links below to study them further:")

                for topic_name, score, mean in weak_topics:
                    import urllib.parse
                    query = urllib.parse.quote(f"{topic_name} study material")
                    st.markdown(
                        f"""
                        <div class='topic-card'>
                            <h5>📉 {topic_name}</h5>
                            <div class='topic-stats'>
                                <span>Your Score: <b style='color:#e74c3c'>{score}</b></span>
                            </div>
                            <p class='topic-summary'>You are lagging in this skill area. We recommend reviewing your course notes or searching online for tutorials to catch up.</p>
                            <a href="https://www.google.com/search?q={query}" target="_blank" class="wiki-link">🔍 Find Study Resources on Google →</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with tab2:
        st.markdown("### 📚 Resources Shared by Faculty")
        resources = get_all_resources()

        if not resources:
            st.info("No resources have been uploaded yet. Check back later.")
        else:
            for res in resources:
                file_path = os.path.join(UPLOAD_DIR, res["filename"])
                with st.container():
                    st.markdown(
                        f"""
                        <div class='resource-card'>
                            <b>📄 {res['original_name']}</b><br>
                            <small>Uploaded by <i>{res['uploader']}</i> on {res['upload_date']}</small><br>
                            {f"<small>{res['description']}</small>" if res['description'] else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            st.download_button(label="⬇️ Download", data=f, file_name=res["original_name"], key=f"dl_{res['id']}")
                    else:
                        st.warning("File not found on server.")
                    st.markdown("---")
