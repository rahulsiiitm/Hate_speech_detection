import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def create_features(corpus, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def save_vectorizer(vectorizer, path='models/vectorizer.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)