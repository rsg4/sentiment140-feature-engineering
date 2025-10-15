# Sentiment Analysis on Sentiment140 with Feature Engineering

**Goal:** Build a stronger sentiment classifier for tweets by combining **n-gram text features** with **VADER sentiment scores**.

## 🔍 Overview
- Dataset: [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/data)  
- Labels: 0 = negative, 2 = neutral, 4 = positive  
- Approach:
  - Clean & preprocess tweets
  - Engineer features: **CountVectorizer n-grams (1–2)** + **VADER compound sentiment score**
  - Train & evaluate a classifier

## 🛠 Tech Stack
- Python, Google Colab  
- scikit-learn, NLTK (VADER), NumPy, SciPy, pandas, matplotlib

## 📂 What’s in this repo
- `Sentiment140.ipynb` — full notebook with preprocessing, feature engineering, training, and evaluation  
- `Assignment 4.html` — static HTML export (optional)  
- `requirements.txt` — minimal dependencies

## 🚀 How to run
```bash
# 1) clone
git clone https://github.com/<your-username>/sentiment140-feature-engineering.git
cd sentiment140-feature-engineering

# 2) install deps (ideally in a virtual env)
pip install -r requirements.txt

# 3) open the notebook
# (Jupyter) 
jupyter notebook Sentiment140.ipynb
# or open in Google Colab by uploading the notebook

> Dataset access: Download the CSV(s) from the Kaggle link and update the notebook path to your local/Drive location. The repo does not include the dataset.



🧪 Method (short)

1. Feature Engineering

N-grams: CountVectorizer(ngram_range=(1,2), max_features=5000)

VADER: SentimentIntensityAnalyzer().polarity_scores(text)['compound']

Concatenate (scipy.sparse.hstack) → X_combined



2. Model

Logistic Regression (baseline & improved with extra features)



3. Metrics

Accuracy, Precision, Recall, F1




✅ Results

Accuracy: Train 0.7796 · Val 0.7772 · Test 0.7766

Balanced performance across classes; improvements from feature enrichment.


📌 Notes

For faster iteration, try a subset of the dataset.

You can swap Logistic Regression with Linear SVM for potential gains.

Future work: POS features, character n-grams, or deep models (LSTM/BERT).


📜 License

MIT — feel free to use and adapt.

🙌 Credits

Dataset: Sentiment140 (Go, Bhayani, Huang)

VADER: Hutto & Gilbert (2014)


---
