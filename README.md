# Gurgaon Real Estate Analytics, Valuation & Recommender System

An end-to-end Machine Learning and Data Science capstone project that analyzes real estate property listings across Gurgaon, performs extensive exploratory data analysis, cleans and engineers rich features, builds high-performance predictive valuation models, and powers an intelligent property recommender system.

---

## 📌 Project Overview

The real estate market is notoriously opaque, dynamic, and non-linear. Property prices depend heavily on multidimensional factors such as location (sector), property type (flat vs. independent house), floor level, built-up vs. carpet area, age of possession, luxury amenities, and proximity to key commercial and transit hubs.

This project delivers a complete data science pipeline to solve these challenges:
1. **Data Preprocessing & Cleaning**: Standardizes messy listing data scraped from major property portals (99acres) into clean, structured records.
2. **Exploratory Data Analysis (EDA)**: Discovers price trends, spatial patterns, and feature correlations via univariate, bivariate, and automated profiling.
3. **Outlier Treatment & Missing Value Imputation**: Implements robust statistical techniques and machine learning imputers (KNN, iterative, group-wise).
4. **Feature Engineering & Selection**: Derives domain-specific indicators (e.g., Luxury Score, Area Ratios, Floor Categories) and prunes redundant features using SHAP, Random Forest importance, and mutual information.
5. **Predictive Modeling**: Compares and tunes multiple regression models (Linear, Tree-based ensembles, XGBoost, LightGBM, CatBoost) to accurately estimate property prices.
6. **Recommender Engine**: Implements a content-based recommendation system that scores similarity across locations, price brackets, and amenities.
7. **Geospatial & Business Insights**: Integrates geospatial coordinates (lat/long) to visualize price heatmaps and generate actionable market insights.

---

## 🗂️ Repository Structure

```text
├── appartments.csv                                        # Apartment listings with society details
├── flats.csv                                              # Raw scraped flat records
├── flats_cleaned.csv                                      # Cleaned flat data
├── houses.csv                                             # Raw scraped house records
├── house_cleaned.csv                                      # Cleaned house data
├── gurgaon_properties.csv                                 # Consolidated flat & house dataset
├── gurgaon_properties_cleaned_v1.csv                      # Level-1 cleaned properties dataset
├── gurgaon_properties_cleaned_v2.csv                      # Level-2 cleaned dataset with standard columns
├── gurgaon_properties_outlier_treated.csv                 # Outlier-filtered property data
├── gurgaon_properties_missing_value_imputation.csv        # Imputed dataset
├── gurgaon_properties_post_feature_selection.csv          # Feature-selected dataset (v1)
├── gurgaon_properties_post_feature_selection_v2.csv       # Final model-ready dataset (v2)
├── latlong.csv                                            # Geographic coordinates of Gurgaon sectors
├── output_report.html                                     # Automated Pandas Profiling EDA report
│
├── Notebooks
│   ├── data-preprocessing-flats.ipynb                    # Step 1: Cleaning flat records
│   ├── data-preprocessing-houses.ipynb                   # Step 2: Cleaning house records
│   ├── merge-flats-and-house.ipynb                       # Step 3: Merging flats and houses
│   ├── data-preprocessing-level-2.ipynb                  # Step 4: Sector & column standardization
│   ├── eda-pandas-profiling.ipynb                        # Step 5: Profiling report generation
│   ├── eda-univariate-analysis.ipynb                     # Step 6: Single-variable distributions
│   ├── eda-multivariate-analysis.ipynb                   # Step 7: Multi-variable & correlation analysis
│   ├── data-visualization.ipynb                          # Step 8: Geospatial & interactive visual analysis
│   ├── outlier-treatment.ipynb                           # Step 9: Outlier detection & capping
│   ├── missing-value-imputation.ipynb                    # Step 10: Missing value imputation strategies
│   ├── feature-engineering.ipynb                         # Step 11: Feature extraction & luxury scoring
│   ├── feature-selection.ipynb                           # Step 12: Feature importance & SHAP analysis
│   ├── feature-selection-and-feature-engineering.ipynb   # Combined feature workflow
│   ├── baseline model.ipynb                              # Step 13: Baseline regression benchmarking
│   ├── model-selection.ipynb                             # Step 14: Model comparison & hyperparameter tuning
│   ├── recommender-system.ipynb                          # Step 15: Content-based property recommender
│   └── insights-module.ipynb                             # Step 16: Analytical reporting & business takeaways
│
├── .gitignore                                             # Git ignore rules for checkpoints and cache
└── README.md                                              # Project documentation
```

---

## ⚙️ Step-by-Step Pipeline

### 1. Data Cleaning & Integration
- Consolidated separate datasets for high-rise flats and independent houses into a unified schema.
- Parsed price units (`Cr` vs `Lakhs`), calculated consistent price-per-square-foot metrics, converted area measurements, and resolved mismatched bedroom/bathroom definitions.

### 2. Exploratory Data Analysis & Spatial Analytics
- Analyzed price distributions, identifying skewness and log-normal behavior.
- Joined sector names with latitude and longitude coordinates (`latlong.csv`) to map property price density across Gurgaon.
- Generated `output_report.html` for deep automated variable inspection.

### 3. Outlier Handling & Missing Value Imputation
- Applied IQR and domain-informed bounds to eliminate erroneous entries (e.g., unrealistic built-up areas or astronomical price-per-sqft anomalies).
- Compared mean, median, KNN, and iterative MICE imputation methods to fill missing values across critical features (`floorNum`, `agePossession`, `super_built_up_area`, etc.).

### 4. Feature Engineering & Selection
- **Luxury Score**: Quantified amenity levels (clubhouse, swimming pool, sports courts, security) into an engineered numerical score.
- **Categorical Binning**: Standardized sector groupings, possession timelines, and floor categories (`Low`, `Mid`, `High`).
- **Feature Importance**: Evaluated features using Random Forest importance, mutual information gain, and SHAP (SHapley Additive exPlanations) values to isolate the most influential predictors.

### 5. Price Prediction Modeling
- Established baselines using Linear Regression, Ridge, and Lasso.
- Evaluated ensemble and gradient-boosted architectures:
  - Random Forest Regressor
  - Extra Trees Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - CatBoost Regressor
- Assessed performance via Cross-Validation using **MAE**, **RMSE**, and **R² Score**, selecting the optimal model architecture through hyperparameter tuning.

### 6. Property Recommender System
- Built a multi-criteria Content-Based Filtering recommender.
- Utilized cosine similarity across:
  - Amenity / facility vectors
  - Price proximity
  - Location and sector proximity
- Allows users to find comparable alternative properties based on their selected preferences.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Git

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <YOUR_GIT_REPO_URL>
   cd "Capstone Project"
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost shap ydata-profiling
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

---

## 📊 Key Highlights & Results

- **Data Integrity**: Over 3,500+ verified Gurgaon properties cleaned, structured, and enriched with sector geolocation.
- **Accurate Valuation**: Non-linear tree-based ensembles (Gradient Boosting, XGBoost, CatBoost) demonstrate strong predictive accuracy over standard linear baselines on real estate valuation.
- **Explainability**: SHAP value analysis clearly reveals how sector location, built-up area, and luxury scores drive property valuation.
- **Personalized Recommendations**: Similarity scoring accurately surfaces top matching alternatives within target budget and amenity criteria.

---

## 📜 License
This project is licensed under the MIT License. Feel free to use and adapt this code for academic, research, and non-commercial projects.
