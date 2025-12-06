import streamlit as st
import pickle
import os
from src.preprocess import clean_text

# --- Page Config ---
st.set_page_config(page_title="Hate Speech Classifier", page_icon="⚖️", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    vect_path = os.path.join('models', 'vectorizer.pkl')
    if not os.path.exists(model_path): return None, None
    with open(model_path, 'rb') as f: model = pickle.load(f)
    with open(vect_path, 'rb') as f: vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("⚖️ Content Toxicity Classifier")
st.markdown("Classifies text into **Hate Speech**, **Offensive Language**, or **Safe**.")

# --- Input ---
user_input = st.text_area("Enter text:", placeholder="Type here...")

if st.button("Analyze") and user_input:
    if model:
        # 1. Preprocess
        cleaned_text = clean_text(user_input)
        
        # 2. Predict
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0] # Get probabilities for [0, 1, 2]

        # 3. Display Results (3-Class Logic)
        st.divider()
        st.subheader("Analysis Result:")
        
        # CLASS 0: HATE SPEECH
        if prediction == 0:
            st.error("🚨 **HATE SPEECH DETECTED**")
            st.write("This text contains hate speech targeting a protected group.")
            st.progress(probs[0], text=f"Confidence: {probs[0]*100:.1f}%")

        # CLASS 1: OFFENSIVE LANGUAGE
        elif prediction == 1:
            st.warning("⚠️ **OFFENSIVE LANGUAGE**")
            st.write("This text contains insults or vulgarity, but is not classified as hate speech.")
            st.progress(probs[1], text=f"Confidence: {probs[1]*100:.1f}%")

        # CLASS 2: NEITHER (SAFE)
        else:
            st.success("✅ **SAFE / NEUTRAL**")
            st.write("This text does not contain hate speech or offensive language.")
            st.progress(probs[2], text=f"Confidence: {probs[2]*100:.1f}%")

        # Optional: Show full breakdown
        with st.expander("See Detailed Probabilities"):
            st.write(f"Hate Speech (0): {probs[0]:.4f}")
            st.write(f"Offensive (1):   {probs[1]:.4f}")
            st.write(f"Neither (2):     {probs[2]:.4f}")