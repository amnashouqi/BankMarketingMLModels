# Bank Marketing Prediction

This repository contains a machine learning project focused on predicting customer subscription to term deposits using the Bank Marketing dataset. The project includes data preprocessing, feature engineering, model training, evaluation, and analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Key Steps](#key-steps)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to develop a machine learning model that predicts whether a bank customer will subscribe to a term deposit. The prediction is based on features such as age, job, education, marital status, and previous marketing campaign outcomes.

---

## Dataset

The dataset used in this project is the **Bank Marketing Dataset**, which contains information on:
- **Input Features**: 
  - Client details (e.g., age, job, marital status, education)
  - Last contact information (e.g., duration, campaign)
  - Socioeconomic context (e.g., employment variation rate, consumer confidence index)
- **Target Variable**: 
  - `y`: Whether the client subscribed to a term deposit (`yes` or `no`).

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `imbalanced-learn`
  - Metrics: `classification_report`, `confusion_matrix`, `accuracy_score`

---

## Key Steps

1. **Data Preprocessing**:
   - Handled missing values and cleaned the dataset.
   - Encoded categorical variables using `LabelEncoder`.
   - Standardized numerical features using `StandardScaler`.

2. **Exploratory Data Analysis**:
   - Examined distributions, correlations, and class imbalance.
   - Visualized data with boxplots.

3. **Feature Engineering**:
   - Dropped low-importance features to reduce overfitting.

4. **Model Training**:
   - Used **Random Forest** and **Neural Networks** for predictions.
   - Balanced the dataset using resampling techniques.

5. **Evaluation**:
   - Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.
   - Performed cross-validation to assess model stability.

---

## Models

### 1. **Random Forest Classifier**
- Ensemble method with decision trees.
- Tuned hyperparameters for optimal performance.

### 2. **Neural Networks**
- Deep learning-based approach using multi-layer perceptrons (MLPs).
- Standardized data for better convergence.

---

## Evaluation Metrics

- **Accuracy**: Overall performance of the model.
- **Precision**: Fraction of true positive predictions among all positive predictions.
- **Recall**: Fraction of true positive predictions among all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Breakdown of true/false positives and negatives.

---

## How to Run the Project

### Prerequisites
- Python 3.8 or above
- Install required libraries using `pip install -r requirements.txt`

### Steps
1. Clone this repository:
   git clone https://github.com/amnashouqi/BankMarketingMLModels.git
   cd BankMarketingMLModels
2. Replace the paths of the data folder downloaded with the hardcoded paths
3. Run the data source code:
   python preprocess.py

## Results

### Random Forest Classifier
- **Accuracy:** 91.27%
- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.97   | 0.95     | 7281    |
| 1     | 0.66      | 0.50   | 0.57     | 954     |

- **Overall Metrics:**
  - Macro Average: Precision: 0.80, Recall: 0.73, F1-Score: 0.76
  - Weighted Average: Precision: 0.90, Recall: 0.91, F1-Score: 0.91

---

### Neural Network (MLP) Classifier
- **Accuracy:** 89.84%
- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.94   | 0.94     | 7281    |
| 1     | 0.56      | 0.62   | 0.59     | 954     |

- **Overall Metrics:**
  - Macro Average: Precision: 0.75, Recall: 0.78, F1-Score: 0.76
  - Weighted Average: Precision: 0.90, Recall: 0.90, F1-Score: 0.90

## Future Improvements

1. Feature Selection:

- Explore additional feature selection methods.
- Remove redundant or less relevant features.
  
2. Hyperparameter Tuning:

- Use techniques like GridSearchCV or Bayesian optimization.
  
3. Model Improvement:

- Experiment with boosting algorithms (e.g., XGBoost, LightGBM).
- Explore deep learning architectures for neural networks.

4. Deploy Model:

- Build a web-based interface for real-time predictions.

## Acknowledgments

- Dataset: UCI Machine Learning Repository - Bank Marketing Dataset.
- Tools: scikit-learn, matplotlib, seaborn, Python.


Feel free to contribute to this project by creating issues or submitting pull requests!
   
