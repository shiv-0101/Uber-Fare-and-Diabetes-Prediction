# 🤖 AIML Assignment — Uber Fare Prediction & Diabetes KNN Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow?logo=googlecolab)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Applied Machine Learning & AI coursework implementing regression and classification pipelines end-to-end.

---

## 📁 Repository Structure

```
├── Assignment1_Uber_Fare_Prediction.ipynb   # Regression: Uber fare prediction
├── Assignment2_KNN_Diabetes.ipynb           # Classification: Diabetes KNN
├── README.md
└── outputs/                                 # Generated plots (auto-created by notebooks)
    ├── boxplot_before.png
    ├── boxplot_after.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── ...
```

---

## 📌 Assignment 1 — Uber Fare Price Prediction

### Problem Statement
Predict the fare amount for an Uber ride given pickup/drop-off coordinates, time, and passenger information.

### Dataset
**[Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)** — 200,000 trips from New York City.

| Column | Description |
|--------|-------------|
| `fare_amount` | 🎯 Target — Fare in USD |
| `pickup_datetime` | Timestamp of the ride |
| `pickup_longitude/latitude` | GPS coordinates of pickup |
| `dropoff_longitude/latitude` | GPS coordinates of drop-off |
| `passenger_count` | Number of passengers |

### Methodology

#### 1. Pre-processing
- Parsed `pickup_datetime` → extracted `hour`, `day_of_week`, `month`, `year`
- Engineered binary features: `is_weekend`, `is_rush_hour`
- Computed **Haversine distance** (great-circle distance in km) between pickup and drop-off
- Applied business logic filters (valid NYC coordinates, fare > $0, etc.)

#### 2. Outlier Detection
Used the **IQR (Interquartile Range) method**:
```
Lower bound = Q1 − 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```
Removed outliers from `fare_amount` and `distance_km`.

#### 3. Correlation Analysis
- Computed Pearson correlation matrix
- Visualised as a heatmap
- **`distance_km`** shows the strongest positive correlation with fare

#### 4. Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline — assumes linear relationship between features and fare |
| **Random Forest Regressor** | Ensemble of 100 trees — captures non-linear patterns |

#### Results

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| MAE    | Higher | **Lower ✅** |
| RMSE   | Higher | **Lower ✅** |
| R²     | Lower  | **Higher ✅** |

**Winner: Random Forest** — captures the non-linear relationship between distance, time, and fare.

---

## 📌 Assignment 2 — K-Nearest Neighbors on Diabetes Dataset

### Problem Statement
Classify whether a patient has diabetes based on diagnostic measurements. Evaluate the model using a full set of classification metrics.

### Dataset
**[Diabetes Dataset (Pima Indians)](https://www.kaggle.com/datasets/abdallamahgoub/diabetes)** — 768 female patients.

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| **Outcome** | 🎯 Target — 1 = Diabetes, 0 = No Diabetes |

### Methodology

#### 1. Pre-processing
- Replaced biologically impossible zero values (Glucose, BMI, etc.) with **class-conditional medians**
- Applied **StandardScaler** — essential for KNN since it is distance-based

#### 2. Finding Optimal K
Used the **Elbow Method** — plotted error rate vs K (1–30) to find the K that minimises error.

#### 3. KNN Algorithm
```
For each test point:
  1. Compute Euclidean distance to all training points
  2. Select K nearest neighbours
  3. Majority vote → predicted class
```

#### 4. Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN) / Total | Overall correctness |
| **Error Rate** | 1 − Accuracy | Overall mistakes |
| **Precision** | TP / (TP+FP) | Of predicted +ve, how many are correct? |
| **Recall** | TP / (TP+FN) | Of actual +ve, how many did we catch? |
| **F1-Score** | 2·P·R / (P+R) | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Area under curve | Overall discrimination ability |

#### Results Summary

| Metric | Score |
|--------|-------|
| Accuracy | ~77–80% |
| Precision | ~70–75% |
| Recall | ~60–68% |
| F1-Score | ~65–72% |
| AUC-ROC | ~80–84% |

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload Notebook** → upload `.ipynb` file
3. Upload your dataset when prompted (or use the built-in synthetic/sklearn loader)
4. Click **Runtime → Run All**

### Option B — Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/aiml-assignment.git
cd aiml-assignment

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook
```

### Datasets
Download from Kaggle and place in the same directory as the notebooks:
- Assignment 1: [uber.csv](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)
- Assignment 2: [diabetes.csv](https://www.kaggle.com/datasets/abdallamahgoub/diabetes)

> **Note:** Both notebooks include fallback data loaders so they run without the Kaggle files (using sklearn's built-in Pima dataset for A2 and synthetic data for A1).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualisation |
| scikit-learn | ML models & evaluation |
| Google Colab | Cloud execution |

---

## 📊 Key Visualisations Generated

**Assignment 1:**
- Boxplots before/after outlier removal
- Fare amount distribution
- Correlation heatmap
- Fare vs Distance scatter plot
- Average fare by hour of day
- Random Forest feature importance
- Actual vs Predicted plots
- Residual plots

**Assignment 2:**
- Feature distributions by outcome class
- Correlation heatmap
- Elbow method (Error Rate vs K)
- Confusion matrix (counts + normalised)
- ROC curve with AUC
- Metrics dashboard
- KNN decision boundary (Glucose vs BMI)

---

## 📤 Push to GitHub

```bash
# First time setup
git init
git add .
git commit -m "Add AIML assignments: Uber fare prediction + Diabetes KNN"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/aiml-assignment.git
git push -u origin main

# Subsequent updates
git add .
git commit -m "Update notebooks"
git push
```

---

## 👤 Author

Devendra Agrawal  
Department of Computer Engineering / AI & ML  
MIT World Peace University | 2025-27

---

## 📄 License

MIT — free to use and modify for educational purposes.
