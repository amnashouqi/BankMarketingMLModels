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
  - Machine Learning: `scikit-learn`, `imbalanced-learn`, `TensorFlow/Keras`  
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
- Deep learning-based approach using multi-layer perceptrons (MLPs) implemented with TensorFlow/Keras.  
- Standardized data for better convergence.  

---

## Evaluation Metrics  

- **Accuracy**: Overall performance of the model.  
- **Precision**: Fraction of true positive predictions among all positive predictions.  
- **Recall**: Fraction of true positive predictions among all actual positives.  
- **F1-Score**: Harmonic mean of precision and recall.  
- **Confusion Matrix**: Breakdown of true/false positives and negatives.  

---

## Results  

### Random Forest Classifier  
- **Accuracy:** 91.40%  
- **Classification Report:**  

| Class | Precision | Recall | F1-Score | Support |  
|-------|-----------|--------|----------|---------|  
| 0     | 0.94      | 0.97   | 0.95     | 7302    |  
| 1     | 0.66      | 0.50   | 0.57     | 936     |  

- **Overall Metrics:**  
  - Macro Average: Precision: 0.81, Recall: 0.74, F1-Score: 0.77  
  - Weighted Average: Precision: 0.91, Recall: 0.92, F1-Score: 0.91  

---

### Neural Network (MLP) Classifier  
- **Accuracy:** 89.99%  
- **Classification Report:**  

| Class | Precision | Recall | F1-Score | Support |  
|-------|-----------|--------|----------|---------|  
| 0     | 0.93      | 0.96   | 0.94     | 7302    |  
| 1     | 0.58      | 0.42   | 0.49     | 939     |  

- **Overall Metrics:**  
  - Macro Average: Precision: 0.76, Recall: 0.69, F1-Score: 0.72  
  - Weighted Average: Precision: 0.89, Recall: 0.90, F1-Score: 0.89 

---

## Future Improvements  

- Explore additional feature engineering techniques to boost model performance.  
- Experiment with other machine learning algorithms like Gradient Boosting, XGBoost, or SVM.  
- Conduct hyperparameter tuning for the Neural Network using grid search or random search.  
- Test the models on additional datasets or real-world banking data.  

---

## Acknowledgments  

- Dataset: UCI Machine Learning Repository - Bank Marketing Dataset.  
- Tools: scikit-learn, TensorFlow, pandas, matplotlib.  

Feel free to contribute to this project by creating issues or submitting pull requests!  :)
