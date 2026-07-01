# Capstone Project: Predictive Analysis for Cardiac Diagnosis — Prioritizing Comprehensive Patient Identification

## 🩺 Problem

This project works with a dataset of health metrics from heart patients, including age, blood pressure, heart rate, and more. The goal is to build a predictive model that accurately identifies individuals with heart disease. Because missing a positive diagnosis carries serious consequences, the primary focus is on achieving high **recall for the positive class**, so that as few true patients as possible are missed.

## 🎯 Objectives

- Explore the dataset to uncover patterns, distributions, and relationships.
- Conduct extensive exploratory data analysis (EDA), including bivariate relationships against the target.
- Preprocess the data:
  - Remove irrelevant features
  - Handle missing values
  - Treat outliers
  - Encode categorical variables
  - Transform skewed features toward a more normal-like distribution
- Build models:
  - Use pipelines for models that require feature scaling
  - Implement and tune KNN, SVM, Decision Tree, and Random Forest classifiers
  - Prioritize high recall for class 1 (presence of heart disease)
- Evaluate and compare model performance using precision, recall, and F1-score.

## 📊 Dataset

The dataset contains 303 patient records and 14 columns, with no missing values. Features include:

- **age** – Age of the patient in years
- **sex** – Gender of the patient
- **cp** – Chest pain type (4 categories)
- **trestbps** – Resting blood pressure (mm Hg)
- **chol** – Serum cholesterol (mg/dl)
- **fbs** – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg** – Resting electrocardiographic results
- **thalach** – Maximum heart rate achieved
- **exang** – Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak** – ST depression induced by exercise relative to rest
- **slope** – Slope of the peak exercise ST segment
- **ca** – Number of major vessels colored by fluoroscopy
- **thal** – Thalium stress test result
- **target** – Heart disease status (0 = no disease, 1 = presence of disease)

## 🔍 Key EDA Insights

- The target classes are fairly balanced: about 54.5% of patients have heart disease and 45.5% do not.
- Maximum heart rate (thalach) shows the strongest visible relationship with heart disease, followed by ST depression (oldpeak) and age.
- Patients with heart disease tend to be slightly younger and achieve higher maximum heart rates during stress tests, while showing lower ST depression on average.
- Resting blood pressure and cholesterol show limited differentiating power between the two classes.
- Outliers were found in trestbps, chol, thalach, and oldpeak using the IQR method. Given the small dataset size, outliers were not removed; instead, Box-Cox-style transformations were applied to reduce their impact, which is especially helpful for distance-based algorithms like KNN and SVM.

## ⚙️ Preprocessing

- Nine numerically-encoded but semantically categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal, target) were converted to categorical type for correct analysis.
- Skewed continuous features were transformed to reduce the influence of outliers and approximate a normal distribution.
- Data was split into training and test sets (80/20) using a stratified split to preserve class balance.
- Feature scaling was applied via pipelines only for models that require it (KNN, SVM), while tree-based models (Decision Tree, Random Forest) were left unscaled since they are scale-invariant.
- All models were tuned using GridSearchCV with stratified cross-validation.

## 🤖 Models and Results

All models were evaluated on the held-out test set. Scores below are rounded to two decimal places.

### Decision Tree
Best hyperparameters: criterion = entropy, max_depth = 2, min_samples_leaf = 1, min_samples_split = 2

- Precision (class 0): 0.80
- Precision (class 1): 0.78
- Recall (class 0): 0.71
- Recall (class 1): 0.85
- F1-score (class 0): 0.75
- F1-score (class 1): 0.81
- Macro avg precision: 0.79
- Macro avg recall: 0.78
- Macro avg F1: 0.78
- Accuracy: 0.79

### Random Forest

- Precision (class 0): 0.84
- Precision (class 1): 0.81
- Recall (class 0): 0.75
- Recall (class 1): 0.88
- F1-score (class 0): 0.79
- F1-score (class 1): 0.84
- Macro avg precision: 0.82
- Macro avg recall: 0.81
- Macro avg F1: 0.82
- Accuracy: 0.82

### K-Nearest Neighbors (KNN)

- Precision (class 0): 0.82
- Precision (class 1): 0.85
- Recall (class 0): 0.82
- Recall (class 1): 0.85
- F1-score (class 0): 0.82
- F1-score (class 1): 0.85
- Macro avg precision: 0.83
- Macro avg recall: 0.83
- Macro avg F1: 0.83
- Accuracy: 0.84

The KNN model showed consistent scores between training and test sets, indicating no overfitting.

### Support Vector Machine (SVM)
Best hyperparameters: C = 0.0011, degree = 2, gamma = scale, kernel = linear

- Precision (class 0): 0.94
- Precision (class 1): 0.71
- Recall (class 0): 0.54
- Recall (class 1): 0.97
- F1-score (class 0): 0.68
- F1-score (class 1): 0.82
- Macro avg precision: 0.82
- Macro avg recall: 0.75
- Macro avg F1: 0.75
- Accuracy: 0.77

The SVM achieved a recall of 0.97 for class 1, meaning it correctly identified almost all patients who actually had heart disease. Its F1-score of 0.82 for class 1 shows this high recall did not come at the cost of excessive false positives.

## ✅ Conclusion

Since missing a true heart disease diagnosis carries serious consequences, recall for class 1 was treated as the most important evaluation metric. Ranked by recall on class 1, the models perform as follows:

1. **SVM** – recall 0.97 (highest recall, but lower precision and accuracy)
2. **Random Forest** – recall 0.88 (strong, well-balanced performance)
3. **KNN** – recall 0.85 (best overall accuracy and balance across metrics)
4. **Decision Tree** – recall 0.85 (lowest overall accuracy)

The SVM model is the strongest choice when the priority is catching as many true heart disease cases as possible, since it minimizes missed diagnoses even at some cost to precision. Random Forest and KNN offer a more balanced trade-off between recall and overall accuracy, making them solid alternatives depending on how much false positives should be minimized.

## 🛠️ Tools and Libraries

- Python, pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (KNN, SVM, Decision Tree, Random Forest, GridSearchCV, StratifiedKFold, pipelines)
- scipy (Box-Cox transformation)

## 📁 Project Structure

```
├── heart.csv                  # Dataset
├── capstone_project.ipynb     # Full notebook: EDA, preprocessing, modeling, evaluation
└── README.md                  # Project overview (this file)
```
