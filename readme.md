# Airbnb Price Prediction Project (NYC 2019)

This project predicts Airbnb listing prices in New York City using machine learning models. The workflow is implemented in `main.ipynb` and includes data preprocessing, exploratory analysis, model training, evaluation, and model saving/loading for future predictions.

## Dataset
- **Source:** `AB_NYC_2019.csv`
- **Description:** Contains NYC Airbnb listing data for 2019, including features like neighbourhood, room type, price, minimum nights, reviews, and availability.

## Workflow Overview

### 1. Data Loading & Exploration
- Load the dataset and inspect its structure.
- Visualize distributions, feature ranges, and relationships (e.g., price vs. minimum nights, neighbourhood group counts).
- Check for missing values and basic statistics.

### 2. Data Preprocessing & Feature Engineering
- Drop unused columns (IDs, names, coordinates, etc.).
- Fill missing values (e.g., `reviews_per_month` with 0).
- Handle outliers in `minimum_nights`, `price`, and `availability_365` using group-based medians and capping.
- Balance categorical features (upsample neighbourhood groups and room types).
- Feature engineering:
  - Create `total_cost` (price × minimum_nights), remove outliers, and log-transform.
  - Normalize availability.
  - Encode categorical columns with `LabelEncoder`.
- Prepare features (`X`) and target (`y`).

### 3. Data Splitting
- Split data into training, validation, and test sets (80/10/10 split).

### 4. Model Training & Evaluation
- **Linear Regression:** Train and evaluate on all splits.
- **Decision Tree:** Train, predict, and evaluate.
- **Random Forest:** Train, predict, evaluate, and analyze feature importance.
- Use metrics: MAE, RMSE, R².
- Compare models visually and in tabular form.

### 5. Model Saving & Loading
- Save trained models and preprocessors (`joblib`).
- Demonstrate loading and testing saved models on the test set.

### 6. Hyperparameter Tuning (Random Forest)
- Tune `n_estimators` for Random Forest using a loop and select the best value based on R².
- Evaluate and compare final models on the real (exponentiated) price scale.

## Files
- `main.ipynb`: Main notebook with all code and analysis.
- `AB_NYC_2019.csv`: Dataset.
- `*_model.joblib`: Saved models (linear regression, decision tree, random forest).
- `scaler.joblib`, `encoder.joblib`: Saved preprocessing objects.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib

## Usage
1. Open `main.ipynb` in Jupyter or VS Code.
2. Run cells sequentially to reproduce the workflow.
3. Saved models can be loaded for future predictions on new data (ensure preprocessing matches training pipeline).

## Results
- All three models are compared using MAE, RMSE, and R².
- Random Forest generally achieves the best performance after tuning.
- Feature importance is visualized for interpretability.

---

*For more details, see the code and comments in `main.ipynb`.*

