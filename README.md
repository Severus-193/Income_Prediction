# Income Classification Using Machine Learning

## Project Overview
This project implements a machine learning pipeline to classify individuals' income levels based on demographic and employment attributes. The model is trained on the **Adult Census Income dataset** and optimized using hyperparameter tuning techniques to improve predictive performance.

## Dataset
The **Adult Census Income dataset** comprises various demographic and employment-related features. The target variable, `income`, is a binary classification label:
- `>50K`: Income exceeds $50,000 per year.
- `<=50K`: Income is $50,000 or less per year.

### Features
- **Numerical Features**: Age, Education (years), Capital Gain, Capital Loss, Hours Per Week, etc.
- **Categorical Features**: Workclass, Education, Marital Status, Occupation, Relationship, Race, Sex, Native Country.

## Project Workflow
### 1. Data Preprocessing
- **Handling Missing Values**:
  - Numerical features: Imputed using **median** values.
  - Categorical features: Imputed using **most frequent** values.
- **Feature Scaling**:
  - Standardize numerical features using **StandardScaler**.
- **Encoding Categorical Variables**:
  - Apply **OneHotEncoder** for nominal categorical features.
  - Use **Label Encoding** where appropriate to reduce dimensionality.

### 2. Model Training
- **Data Splitting**: 
  - Dataset is divided into **80% training** and **20% test** subsets.
- **Machine Learning Model**: 
  - Implement **Random Forest Classifier** due to its robustness and feature importance insights.
- **Hyperparameter Optimization**:
  - Use **GridSearchCV** to tune hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
  - Select optimal parameters based on cross-validation performance.

- **Feature Importance Analysis**:
  - Extract feature importance scores from **Random Forest Classifier**.
  - Visualize feature rankings using **Seaborn** bar plots.

## Installation & Setup
### Prerequisites
Ensure Python (>=3.8) and the required dependencies are installed:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_directory>
   ```
2. Execute the script:
   ```sh
   python income_prediction.py
   ```

## Results & Insights
- The model achieves a high **classification accuracy**, effectively distinguishing between income classes.
- **Feature Importance Analysis** identifies key factors influencing income, such as **education level, hours worked per week, and capital gains**.
- Model interpretability is enhanced through **feature visualization techniques**.


