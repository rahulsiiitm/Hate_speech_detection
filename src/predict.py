import pickle
import os
import sys

# Add 'src' to path to import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

def load_artifacts():
    """Loads the saved model and vectorizer."""
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    vect_path = os.path.join('models', 'vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        print("Error: Model files not found. Run 'src/train.py' first!")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def predict_text(text, model, vectorizer):
    # 1. Clean
    cleaned_text = clean_text(text)
    
    # 2. Vectorize
    features = vectorizer.transform([cleaned_text])
    
    # 3. Predict
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    return pred, probs

if __name__ == "__main__":
    model, vectorizer = load_artifacts()
    
    labels = {0: "HATE SPEECH 🔴", 1: "OFFENSIVE 🟠", 2: "SAFE 🟢"}
    
    if model:
        print("\n--- Hate Speech Detector CLI ---")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("Enter text: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            pred, probs = predict_text(user_input, model, vectorizer)
            
            print(f"Result: {labels[pred]}")
            print(f"Confidence: {probs[pred]*100:.1f}%\n")