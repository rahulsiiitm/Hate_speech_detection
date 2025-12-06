import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features import create_features, save_vectorizer

def load_processed_data(filepath):
    print(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath)
    df.dropna(subset=['cleaned_text'], inplace=True)
    
    # 0 = Hate, 1 = Offensive, 2 = Neither
    if 'class' in df.columns:
        df['label'] = df['class']
    
    return df

def main():
    data_path = os.path.join('data', 'processed', 'processed_data.csv')
    if not os.path.exists(data_path):
        print("Error: Run 'src/preprocess.py' first.")
        return

    df = load_processed_data(data_path)

    print("Extracting features...")
    # Increase max_features to capture more subtle words
    X, vectorizer = create_features(df['cleaned_text'], max_features=10000) 
    y = df['label']
    save_vectorizer(vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression (Balanced)...")
    # class_weight='balanced' tells the model to pay more attention to rare classes (Hate/Neutral)
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    with open('models/hate_speech_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Success! Balanced model saved.")

if __name__ == "__main__":
    main()