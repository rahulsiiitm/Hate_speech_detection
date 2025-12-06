# Hate Speech Detection System

##  Project Overview
This project is a Machine Learning pipeline designed to detect hate speech in textual data. It utilizes Natural Language Processing (NLP) techniques for data preprocessing and feature extraction, followed by a **Multinomial Naive Bayes** classifier.

The system is trained to distinguish between:
* **Hate Speech**
* **Non-Hate Speech** (including offensive language and neutral text)

##  Folder Structure
```text
Hate-Speech-Detection/
│
├── data/
│   ├── raw/                 # Contains 'labeled_data.csv'
│
├── src/                     # Source code
│   ├── preprocess.py        # Text cleaning logic (URLs, stopwords)
│   ├── features.py          # Feature extraction (CountVectorizer)
│   ├── train.py             # Model training and evaluation script
│   └── __init__.py          # Package initialization
│
├── models/                  # Saved artifacts
│   ├── hate_speech_model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt         # Dependencies
├── main.py                  # Entry point for the application
└── README.md                # Project documentation
```

## Installation

1. **Clone the repository or navigate to the project directory:**
   ```bash
   cd Hate_speech_detection
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the hate speech detection model on the labeled dataset:

```bash
python src/train.py
```

This script will:
- Load the dataset from `data/raw/labeled_data.csv`
- Clean and preprocess the text
- Extract features using CountVectorizer
- Train a Multinomial Naive Bayes classifier
- Evaluate the model on a test set
- Save the trained model and vectorizer to the `models/` directory

### Model Performance
The model outputs:
- **Accuracy Score**: Overall accuracy on the test set
- **Confusion Matrix**: True positives, true negatives, false positives, and false negatives
- **Classification Report**: Precision, recall, and F1-score (available in extended version)

## Dataset

The project uses the **labeled_data.csv** dataset containing:
- `tweet`: The text content of tweets
- `class`: Label indicating the type of content
  - `0`: Hate Speech
  - `1`: Offensive Language
  - `2`: Neutral/Neither

The dataset is preprocessed to create a binary classification problem:
- **Class 1**: Hate Speech
- **Class 0**: Non-Hate Speech (offensive language + neutral)

## Architecture

### 1. **Preprocessing (preprocess.py)**
- Converts text to lowercase
- Removes URLs using regex patterns
- Removes punctuation
- Filters out English stopwords (e.g., "the", "is", "and")

### 2. **Feature Extraction (features.py)**
- Uses `CountVectorizer` from scikit-learn
- Converts text into numerical feature vectors
- Maximum of 5000 features
- Saves the vectorizer for consistent transformation during inference

### 3. **Model Training (train.py)**
- Splits data into 80% training and 20% testing
- Trains a **Multinomial Naive Bayes** classifier
- Evaluates performance on test set
- Persists the model and vectorizer as pickle files

## Dependencies

- **pandas**: Data manipulation and loading
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and metrics
- **nltk**: Natural language processing and stopwords
- **matplotlib & seaborn**: Data visualization (optional)

## Future Enhancements

- [ ] Implement model with TF-IDF instead of CountVectorizer
- [ ] Add support for deep learning models (e.g., LSTM, BERT)
- [ ] Create a Flask/FastAPI web application for real-time predictions
- [ ] Implement cross-validation for better model evaluation
- [ ] Add data visualization and exploratory data analysis
- [ ] Deploy model as a REST API
- [ ] Support multiple languages

## License

This project is part of the SparkIIT internship program.

## Author

Rahul Sharma