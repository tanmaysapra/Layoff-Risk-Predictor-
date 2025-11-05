# Layoff Risk Predictor

A Machine Learning project that predicts **layoff percentages** of companies using data from **[Layoffs.fyi](https://layoffs.fyi/)**.
This project compares multiple regression models — from simple linear regression to advanced ensemble techniques — to identify key economic and organizational factors influencing workforce downsizing.

---

## Repository

🔗 https://github.com/tanmaysapra/Layoff-Risk-Predictor-

## Overview

### Objective

To predict **company layoff percentages** based on multiple parameters such as:

* Company funding stage
* Industry sector
* Geographical location
* Funds raised
* Year and month of operation

### Motivation

Layoffs have surged globally, creating economic instability and uncertainty. This project aims to assist organizations, researchers, and policymakers in **proactively identifying at-risk companies or sectors** through data-driven modeling.

---

## Methodology

### 1. **Data Collection**

* Source: [Layoffs.fyi Dataset (2020–2024)](https://layoffs.fyi/)
* Format: CSV
* Size: ~3,000 records post-cleaning

### 2. **Preprocessing**

* Removed duplicates and handled missing values
* Encoded categorical features (*industry, stage, country*)
* Scaled numeric attributes (*funds_raised*)
* Chronological data split: Training (≤2023), Testing (>2023)

### 3. **Modeling**

Implemented and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression
* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* CatBoost Regressor
* XGBoost Regressor

Each model was tuned using cross-validation for hyperparameter optimization.

---

## Visualizations

The repository includes plots for:

* **Actual vs Predicted Layoff %**
* **Residual and Q–Q Plots**
* **Feature Importance (Tree-Based Models)**
* **MAE vs Hyperparameter Curves**
* **Decision Tree Visualization**

Example:

```
Actual vs Predicted Layoff % (CatBoost)
Feature Importance (Stage, Industry, Funds Raised)
Residuals vs Predicted (Decision Tree)
MAE vs Alpha (Ridge/Lasso)
```

---

## Model Comparison

| Model             | Category         | MAE ↓     | R² ↑     | Key Insight              |
| ----------------- | ---------------- | --------- | -------- | ------------------------ |
| Linear Regression | Linear           | 0.058     | 0.41     | Baseline model           |
| Ridge             | Linear (L2)      | 0.051     | 0.47     | Stable coefficients      |
| Lasso             | Linear (L1)      | 0.049     | 0.50     | Sparse feature selection |
| Elastic Net       | Hybrid           | 0.047     | 0.52     | Balanced regularization  |
| KNN               | Non-Parametric   | 0.045     | 0.56     | Captures local trends    |
| Decision Tree     | Tree             | 0.042     | 0.60     | Interpretable splits     |
| Random Forest     | Ensemble         | 0.037     | 0.68     | Strong generalization    |
| **CatBoost**      | Boosted Ensemble | **0.034** | **0.72** | Best overall performer   |

---

## Key Takeaways

* **Stage**, **industry**, and **funds raised** are the most predictive factors for layoffs.
* **CatBoost Regressor** offered the best combination of accuracy and interpretability.
* Ensemble models outperform single regressors due to non-linear layoff behavior.

---

## Future Work

* Integrate **macroeconomic indicators** (inflation, GDP, unemployment)
* Deploy **interactive dashboards** for real-time layoff risk prediction
* Explore **neural architectures (RNN, Transformer)** for temporal layoff forecasting

---

## Tech Stack

* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `catboost`
* **Tools:** Jupyter Notebook, GitHub, Visual Studio Code

---

## How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/praventhegenius/Layoff-Risk-Predictor.git
   cd Layoff-Risk-Predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook Layoff_Risk_Predictor.ipynb
   ```

4. **View results**

   * Visualizations under `/visuals`
   * Model comparison outputs in terminal or notebook cells

---

## Author



**Tanmay Sapra** : [GitHub Profile – tanmaysapra](https://github.com/tanmaysapra)

**Pravenraam Shankar** : [GitHub Profile – praventhegenius](https://github.com/praventhegenius)


---

## 🏁 License

This project is open-sourced under the **MIT License** — feel free to use, modify, and cite with attribution.

---
