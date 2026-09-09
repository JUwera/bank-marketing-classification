# Random-Forest-Classifier-for-Bank-Marketing-Prediction
## Overview
This project demonstrates the development and evaluation of a machine learning model, focusing on ensemble classification techniques and data integrity. The objective was to build a Random Forest Classifier using the UCI Bank Marketing dataset, resolve critical operational issues like data leakage, handle class imbalance, and extract feature importance to predict customer subscription behavior.

## Objectives
* **Data Pre-processing:** Implement missing value imputation (mode imputation) and categorical variable encoding.
* **Pipeline Optimization:** Evaluate model performance before and after isolating the call duration feature to prevent data leakage.
* **Hyperparameter Tuning:** Leverage Out-of-Bag (OOB) evaluation to systematically tune model estimators and tree depth.
* **Imbalance Management:** Implement cost-sensitive learning to adjust training sample distribution weights.
* **Portfolio Showcase:** Deliver a complete, reproducible machine learning pipeline hosted entirely within a single interactive notebook workspace.

## Technologies Used
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Pipelines:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Version Control:** GitHub

## Experimental Framework
The machine learning pipeline was designed using a standalone Jupyter Notebook framework and evaluates the following model architectures:
* **Logistic Regression:** Implemented with balanced class weights as a continuous probability baseline.
* **Decision Tree Classifier:** Built to monitor baseline hierarchical branching and overfitting tendencies on dense datasets.
* **Random Forest Classifier:** Developed as the primary ensemble learning engine using automated bootstrap sample bagging.

## Design Considerations
* **Data Leakage Mitigation:** Discarding the phone call `duration` parameter to form a realistic operational predictive platform.
* **Missing Feature Handling:** Mapping and replacing missing categorical string tags coded under the `"unknown"` label.
* **Class Vector Balancing:** Applying explicit algorithmic weights (`class_weight='balanced'`) to manage the highly skewed data distribution.
* **Efficient Validation Strategy:** Utilizing Out-of-Bag (OOB) scores as a validation set variant to maximize training sample efficiency.

## Key Performance Results
* **Optimal Architecture Parameters:** 300 estimators with unrestricted tree depth (None) yielding a top OOB validation score of 0.9111.
* **Baseline Comparative Benchmarks (With Leakage):** Logistic Regression (86.43% Accuracy) | Decision Tree (85.55% Accuracy) | Random Forest (91.33% Accuracy).
* **Primary Demographic Indicators:** Identification of client age, campaign contacts, and specific economic index shifts as top feature importance factors.

## Key Learnings
* **Ensemble Learning Mastery:** Developed an understanding of how Random Forest models utilize multiple decision trees to generalize on unexplored datasets.
* **Operational Risk Management:** Gained hands-on experience identifying and rectifying data leakage to align machine learning parameters with real-world business constraints.
* **Statistical Imbalance Processing:** Learned how to deploy cost-sensitive learning matrices to prevent minority class misclassification.
* **Visual Exploration Synthesis:** Practiced using behavioral histograms and feature importance plots to map out actionable insights.

## Future Improvements
* **Advanced Imbalance Oversampling:** Integrate SMOTE (Synthetic Minority Over-sampling Technique) pipelines to further balance the target data.
* **Model Explainability Frameworks:** Implement SHAP (SHapley Additive exPlanations) or LIME to make individual tree splits completely transparent.
* **Alternative Boosting Benchmarks:** Test the finalized preprocessing architecture against gradient boosting alternatives like XGBoost or LightGBM.

## Credits
This project was developed as a comprehensive application of direct marketing analytics and ensemble learning concepts for predictive classification.

* **Developer:** Joanne Uwera
* **Dataset Reference:** Moro et al., 2014 (UCI Machine Learning Repository)
