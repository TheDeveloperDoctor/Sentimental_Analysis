# 🧠 Sentiment Analysis ML Pipeline

This project is a full machine learning pipeline that predicts the sentiment of social media text (positive, negative, or neutral). It includes classical ML (Random Forest, Logistic Regression), deep learning (TensorFlow DNN), and deployment-ready preprocessing using `ColumnTransformer`.

---

## 📦 Features

* Cleaned and preprocessed a real-world Twitter dataset
* Applied **TF-IDF vectorization** to textual data
* Encoded categorical features using **OneHotEncoder**
* Trained and compared:

  * ✅ **Random Forest Classifier**
  * ✅ **Logistic Regression**
  * ✅ **TensorFlow Deep Neural Network (DNN)**
* Exported model and pipeline using `pickle` for deployment
* Included helper function to make live predictions
* Visualized sentiment distribution and training progress

---

## 📊 Dataset Overview

| Feature                       | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `text`                        | Raw tweet text                              |
| `sentiment`                   | Target class: positive / neutral / negative |
| `age_group`                   | Age group of the tweet's author             |
| `time_of_tweet`               | When the tweet was posted                   |
| `Country`, `population`, etc. | Dropped from final model                    |

> 📌 Total rows after cleaning: **3534**
>
> 🧼 Cleaning included: removing URLs, punctuation, single characters, and stopwords.

---

## 🏗️ Pipeline Architecture

```
[text, time_of_tweet, age_group]
            │
     ┌────▼────┐
     │ ColumnTransformer │
     └────▲────┘
TF-IDF on 'clean_text' + OHE on categorical columns
            │
     ┌────▼────┐
     │ ML Model (RF / DNN) │
     └────▲────┘
    Predict Sentiment
```

---

## 🧠 Model Performance

| Model               | Accuracy                  | Notes                      |
| ------------------- | ------------------------- | -------------------------- |
| Logistic Regression | 64.21%                    | Balanced but basic         |
| Random Forest       | 64.36%                    | Stronger on positive class |
| TensorFlow DNN      | 98.64% train / 59.01% val | Overfitting observed       |

### 🚪 Classification Report (Random Forest)

```
              precision    recall  f1-score
negative       0.70        0.44     0.54
neutral        0.57        0.77     0.65
positive       0.76        0.67     0.71
accuracy                             0.64
```

---

## 🔮 Predict Sentiment Example

```python
from predict import predict_sentiment

print(predict_sentiment("I love this product! It's amazing."))
# Output: positive

print(predict_sentiment("This is the worst experience I've ever had."))
# Output: negative
```

---

## 🚀 Deployment

Models and transformers are saved using `pickle`:

* `website/RandomForest.pkl`
* `website/data_preprocessor.pkl`

These can be loaded in a web API (e.g., FastAPI or Flask) for live predictions.

---

## 🧰 Tech Stack

* **Pandas, NumPy** – Data cleaning and manipulation
* **NLTK** – Stopword removal and lemmatization
* **scikit-learn** – TF-IDF, OneHotEncoder, Random Forest, Logistic Regression
* **TensorFlow / Keras** – Deep learning model
* **Matplotlib / Seaborn** – Visualization
* **Pickle** – Saving models for reuse

---

## 📂 Project Structure

```
.
├── data/                      # Dataset (test.csv, cleaned CSVs)
├── website/
│   ├── RandomForest.pkl       # Final model
│   └── data_preprocessor.pkl  # ColumnTransformer
├── predict.py                 # predict_sentiment() function
├── SentimentAnalysis.ipynb    # Full ML pipeline notebook
├── README.md                  # This file
```

---

## 📌 Future Work

* [ ] Deploy the model via **FastAPI**
* [ ] Build a frontend using **Streamlit** or **Gradio**
* [ ] Add support for multilingual sentiment analysis
* [ ] Integrate with live Twitter API

---

## 🙌 Acknowledgments

This project was developed as part of my AI/ML portfolio. Built by [Haris Ahmed](https://www.linkedin.com/in/haris-ahmed-785480257/).
