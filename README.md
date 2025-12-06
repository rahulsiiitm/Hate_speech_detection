
# Hate Speech & Toxicity Detection System

## Project Overview

This project is a Machine Learning application designed to detect and classify toxic content in text. Instead of simple binary classification, it distinguishes between **Hate Speech**, **Offensive Language**, and **Safe/Neutral** content.

The system handles class imbalance (where hate speech samples are comparatively rare) using a **Logistic Regression model with balanced class weights**, improving minority class detection.

## Key Features

- Multi-Class Classification:
  - Hate Speech: Targeted attacks based on attributes such as race, gender, religion, etc.
  - Offensive Language: General insults, profanity, or vulgar expressions.
  - Safe: Neutral or non-toxic text.
- Class Imbalance Handling using `class_weight='balanced'`
- Interactive Web Interface using Streamlit
- Command-Line Interface (CLI) for quick predictions
- Modular and Production-Ready Code Structure

## Folder Structure

```
Hate-Speech-Detection/
│
├── data/
│   ├── raw/                 # Original labeled_data.csv (Davidson et al.)
│   └── processed/           # Cleaned and processed datasets
│
├── notebooks/
│   ├── 01_Exploration.ipynb # Exploratory data analysis
│   └── 02_Evaluation.ipynb  # Confusion matrix and metrics
│
├── src/
│   ├── preprocess.py        # Text cleaning and preprocessing
│   ├── features.py          # Feature extraction logic (TF-IDF /CountVectorizer)
│   ├── train.py             # Model training pipeline
│   └── predict.py           # CLI inference script
│
├── models/
│   ├── hate_speech_model.pkl
│   └── vectorizer.pkl
│
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd Hate-Speech-Detection
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the Hate Speech and Offensive Language Dataset from Kaggle and place  
`labeled_data.csv` inside:

```
data/raw/
```

## How to Run

### 1. Data Preprocessing

Clean and prepare the text data for training.

```bash
python src/preprocess.py
```

### 2. Train the Model

Train the Logistic Regression model. The trained model and vectorizer will be saved in the `models/` directory.

```bash
python src/train.py
```

### 3. Run the Web Application

Launch the Streamlit web interface.

```bash
streamlit run app.py
```


## Model Details

- Algorithm: Logistic Regression
- Class Weights: Balanced
- Feature Extraction: CountVectorizer or TF-IDF (configurable)
- Evaluation Metrics:
  - Offensive Language: High precision and recall (majority class)
  - Hate Speech: Optimized for recall to reduce false negatives

## Tech Stack

- Language: Python 3.x
- Machine Learning: Scikit-learn
- NLP: NLTK
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Deployment/UI: Streamlit

## License

This project is open-source and licensed under the MIT License.
