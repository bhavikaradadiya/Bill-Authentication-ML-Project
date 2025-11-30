# Bill Authentication — Machine Learning Project
This project applies multiple Machine Learning algorithms on the Bill Authentication Dataset, which contains statistical features extracted from images of banknotes. The goal is to classify banknotes as genuine (0) or fake (1) using various ML techniques.

# 📝 Project Overview

The dataset contains the following features:
| Feature      | Description                               |
| ------------ | ----------------------------------------- |
| **Variance** | Variance of the wavelet-transformed image |
| **Skewness** | Measure of asymmetry                      |
| **Curtosis** | Sharpness of the peak                     |
| **Entropy**  | Randomness of the image                   |
| **Class**    | 0 = Real note, 1 = Fake note              |

Using these features, the project builds multiple ML models to understand performance across algorithms.

# 🔧 1. Data Cleaning

✔ Load dataset
✔ Check missing values
✔ Detect & remove duplicate rows
✔ Validate datatypes
✔ Dataset reduced from 1372 → 1348 rows after removing duplicates

# 🌳 2. Decision Tree Classification

Split dataset using 70:30 ratio

Train a DecisionTreeClassifier

Evaluate using:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Accuracy Achieved: 98.77%
Confusion matrix showed very strong model performance with low false positives and false negatives

# 🔵 3. K-Means Clustering

Because clustering is unsupervised:

Standardization performed using StandardScaler()

K-Means fitted on standardized feature data

Cluster labels added back to the dataset

Visual clustering plot generated

Centroids displayed

Clustering Accuracy (comparing to actual labels): 43.69%, meaning K-Means is not suitable for this dataset’s structure

# 🌲 4. Classification Evaluation (Random Forest)

Another model was trained for performance comparison.

Metrics computed:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

ROC Curve

Random Forest Accuracy: 91.01%
Achieved excellent recall and F1-score values (close to 1.0) 

# 📈 5. Linear Regression Evaluation

Even though linear regression is not ideal for classification, it was tested for educational purposes.

Computed evaluations:

MAE = 0.13

MSE = 0.03

RMSE = 0.18

R² = 0.88

Boxplots were generated to show:

Actual vs predicted values

Distribution of residuals

# 🧠 Conclusion

The dataset is well-suited for binary classification.

Decision Trees performed best with 98.77% accuracy.

Random Forest also delivered strong generalization.

K-Means Clustering showed poor performance, confirming that this dataset is not suitable for unsupervised classification.

Linear Regression, despite not being intended for classification tasks, produced reasonable regression metrics for educational comparison.

# How to Run

### 1. Clone the repository:

git clone https://github.com/bhavikaradadiya/Bill-Authentication-ML-Project
cd bill-authentication-ml

### 2. Install dependencies:

pip install -r requirements.txt

### 3. Run the notebook or individual Python scripts:

colab notebook
