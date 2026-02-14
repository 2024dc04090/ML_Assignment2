## a. Problem Statement

Implement and compare six machine learning classification models on a dataset with at least 12 features and 500 instances. Build an interactive Streamlit web application to demonstrate model performance, calculate six evaluation metrics for each model, and deploy the application on Streamlit Community Cloud.

## b. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Download:** https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

**Dataset Characteristics:**
- Instances: 11,162
- Features: 16 input features + 1 target
- Target Variable: deposit (yes/no - term deposit subscription)
- Problem Type: Binary Classification
- Class Distribution: Imbalanced (88% No, 12% Yes)

**Features:**
1. age - Age of client (numeric)
2. job - Type of job (categorical)
3. marital - Marital status (categorical)
4. education - Education level (categorical)
5. default - Has credit in default? (categorical)
6. balance - Average yearly balance in euros (numeric)
7. housing - Has housing loan? (categorical)
8. loan - Has personal loan? (categorical)
9. contact - Contact communication type (categorical)
10. day - Last contact day of month (numeric)
11. month - Last contact month (categorical)
12. duration - Last contact duration in seconds (numeric)
13. campaign - Number of contacts performed (numeric)
14. pdays - Days since last contact (numeric)
15. previous - Number of previous contacts (numeric)
16. poutcome - Outcome of previous campaign (categorical)

**Target:** deposit - Client subscribed to term deposit (yes/no)

---

## c. Models Used

### Model Comparison Table

|| ML Model Name           || Accuracy  || AUC    || Precision || Recall || F1     || MCC    ||
||=========================||===========||========||===========||========||========||========||
||Logistic Regression      || 0.7971    || 0.8729 || 0.7957    || 0.7694 || 0.7823 || 0.5928 ||
||-------------------------||-----------||--------||-----------||--------||--------||--------||
||Decision Tree            || 0.8200    || 0.8470 || 0.7987    || 0.8289 || 0.8135 || 0.6400 ||
||-------------------------||-----------||--------||-----------||--------||--------||--------||
||kNN                      || 0.7900    || 0.8520 || 0.7966    || 0.7476 || 0.7713 || 0.5785 ||
||-------------------------||-----------||--------||-----------||--------||--------||--------||
||Naive Bayes              || 0.7546    || 0.8088 || 0.7150    || 0.8015 || 0.7558 || 0.5141 ||
||-------------------------||-----------||--------||-----------||--------||--------||--------||
||Random Forest (Ensemble) || 0.8397    || 0.9127 || 0.8125    || 0.8601 || 0.8356 || 0.6805 ||
||-------------------------||-----------||--------||-----------||--------||--------||--------||
||XGBoost (Ensemble)       || 0.8536    || 0.9239 || 0.8266    || 0.8743 || 0.8498 || 0.7082 ||


### Model Observations
||  ML Model Name              || Observation about model performance                         ||
||=============================||=============================================================||
|| Logistic Regression         || Provides a solid baseline with 79.71% accuracy and good     ||
||                             || AUC (0.873). Shows balanced precision-recall                ||
||                             || tradeoff (0.796/0.769). Limited by linear decision          ||
||                             || boundaries but offers excellent interpretability and fast   ||
||                             || training. MCC of 0.593 indicates moderate predictive power. ||
||                             || Best suited for baseline modeling and scenarios requiring   || 
||                             || model interpretability.                                     ||
||-----------------------------||-------------------------------------------------------------||
|| Decision Tree               || Achieves 82.00% accuracy with the highest recall (0.829)    || 
||                             || among individual models, making it excellent for identifying|| 
||                             || positive cases. Captures non-linear patterns effectively    || 
||                             || with interpretable tree structure. The max depth constraint ||
||                             || prevents overfitting but AUC (0.847) is lower than ensemble ||
||                             || methods. Good for understanding feature importance and      ||
||                             || decision paths.                                             ||
||-----------------------------||-------------------------------------------------------------||
|| kNN                         || Instance-based learning achieves 79.00% accuracy with       ||
||                             || highest precision (0.797) but lowest recall (0.748),        ||
||                             || indicating conservative predictions. Performance is         ||
||                             || sensitive to k parameter (k=5) and feature scaling.         ||
||                             || Computationally expensive for large datasets. AUC of 0.852  ||
||                             || shows decent ranking ability. Suitable for smaller datasets ||
||                             || where similar instances should have similar predictions.    ||
||-----------------------------||-------------------------------------------------------------||
|| Naive Bayes                 || Despite strong independence assumptions, achieves fastest   ||
||                             || training with 75.46% accuracy. Shows lowest                 ||
||                             || precision (0.715) but high recall (0.802), suitable for     ||
||                             || minimizing false negatives. The probabilistic approach works||
||                             || well with continuous features. AUC of 0.809 is respectable. ||
||                             || Ideal when speed is priority and some accuracy can          ||
||                             || sacrificed.                                                 ||
||-----------------------------||-------------------------------------------------------------||
|| Random Forest (Ensemble)    || Strong ensemble performance with 83.97% accuracy through 100|| 
||                             || decision trees. Excellent AUC (0.913) demonstrates superior ||
||                             || class discrimination. Achieves balanced precision (0.813)   ||
||                             || and recall (0.860). Reduces overfitting while providing     ||
||                             || feature importance insights. MCC of 0.681 shows strong      || 
||                             || correlation. Great balance of accuracy, robustness, and     ||
||                             || interpretability for production use.                        ||
||-----------------------------||-------------------------------------------------------------||
|| XGBoost (Ensemble)          || Best overall performer with 85.36% accuracy and outstanding ||
||                             || AUC (0.924). Sequential gradient boosting achieves highest  ||
||                             || precision (0.827), recall (0.874), and F1 score (0.850).    ||
||                             || MCC of 0.708 confirms strongest predictive power. Built-in  ||
||                             || regularization handles class imbalance effectively. Requires||
||                             || careful tuning but delivers state-of-the-art performance.   ||
||                             || Recommended for production deployment.                      ||

---

## Project Structure

```
Assignment2/
├── data/
│   └── bank.csv
├── models/
│   └── all_models.pkl
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── app.py
├── styles.py
├── constants.py
├── requirements.txt
└── README.md
```

---

## Installation and Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
cd data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
unzip bank.zip
```

### Train Models
```bash
python src/model_training.py
```

### Run Application
```bash
streamlit run app.py
```

---

## Streamlit App Features

1. **Dataset Upload:** Upload test data in CSV format
2. **Model Selection:** Dropdown to select from 6 trained models
3. **Evaluation Metrics:** Display all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix:** Visual representation of predictions vs actual
5. **Classification Report:** Detailed per-class metrics

---

## Technologies Used

- Python 3.8+
- scikit-learn 1.4.0
- XGBoost 2.0.3
- Streamlit 1.31.0
- Pandas 2.0.3
- NumPy 1.24.3
- Matplotlib 3.7.2
- Seaborn 0.13.1

---

## Deployment

**Live App:** [Your Streamlit App URL]  
**GitHub Repository:** [Your GitHub Repository URL]

Deploy on Streamlit Cloud:
1. Push code to GitHub
2. Visit https://streamlit.io/cloud
3. Sign in and select repository
4. Deploy app.py

---

## Author

Kotha Chandralekha
2024DC04090 
M.Tech (DSE)