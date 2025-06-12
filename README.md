
# 🏠 Real Estate Price Prediction

A machine learning project that predicts house prices based on various property features using advanced regression models. This end-to-end pipeline includes data cleaning, visualization, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 📌 Project Overview

This project aims to build predictive models that estimate the prices of residential properties using structured data. By leveraging various machine learning techniques, we analyze and model the key factors affecting housing prices.

---

## ✨ Key Features

- Exploratory Data Analysis (EDA) with visual insights  
- Data preprocessing and feature engineering  
- Handling skewness and outliers  
- Training multiple regression models:  
  - Linear Regression  
  - Random Forest  
  - XGBoost  
  - AdaBoost  
- Hyperparameter tuning (XGBoost)  
- Feature importance analysis  
- Model comparison and evaluation  
- Model export and prediction on new data

---

## 🧰 Tech Stack

- **Python**  
- **Pandas, NumPy, Seaborn, Matplotlib** – for data wrangling and visualization  
- **Scikit-learn** – for modeling and preprocessing  
- **XGBoost, RandomForest, AdaBoost** – ensemble models  
- **Joblib** – to save trained models  
- *(Optional)* **Streamlit** – for a simple demo interface (not included yet)

---

## 📁 Project Structure

```

real-estate-price-prediction/
│
├── data/
│   └── kc_house_data.csv
│
├── notebooks/
│   └── 01_king_county_house_price_modeling.ipynb
│
├── src/
│   └── models/
│       └── tuned_xgb_model.pkl
│
├── tuned_xgb_predictions.csv
├── requirements.txt
├── README.md
└── .gitignore


```

---

## ⚙️ How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/real-estate-price-prediction.git
   cd real-estate-price-prediction
```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in order (inside `notebooks/`) to follow the full pipeline.

4. To load the final model and make predictions:

   ```python
   import joblib
   model = joblib.load("src/models/tuned_xgb_model.pkl")
   predictions = model.predict(X_new)
   ```

---

## 📊 Results & Insights

* 📈 **Best model:** Tuned XGBoost

* 📉 **Performance on test set:**

  * **MAE:** \~0.12
  * **RMSE:** \~0.16
  * **R² Score:** \~0.91

* 🧠 **Top important features:**

  * `sqft_living`, `grade`, `lat`, `bathrooms`, `view`, `waterfront`, etc.

---

## 🙌 Author

**Yahia Zakaria**
Machine Learning Engineer & Data Science Enthusiast
[LinkedIn](https://www.linkedin.com/in/yahia-zakaria-a27384213/) | [GitHub](https://github.com/Yahia20)

---

## 📌 License

This project is open-source and available under the [MIT License](LICENSE).

```

