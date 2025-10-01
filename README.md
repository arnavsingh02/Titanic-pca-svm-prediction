This repository demonstrates a robust and modular approach to solving the classic Titanic survival prediction problem using custom machine learning algorithms and pipeline engineering. The notebook implements both Support Vector Machine (SVM) and Logistic Regression from scratch, introduces a custom Principal Component Analysis (PCA) routine for feature reduction and variance analysis, and emphasizes reproducible, research-quality training and evaluation.

Dataset
Source: Kaggle Titanic: Machine Learning from Disaster (CSV file required, e.g. titanic.csv)

Features Used: Sex, Age, Pclass, Embarked, SibSp, Parch, Fare

Dropped Features: Name, PassengerId, Ticket, Cabin

Target Variable: Survived (binary classification)

Repository Features
Custom SVM Implementation: Fully coded support vector classifier using primal form with learning rate, regularization, and iterative updates.

Custom Logistic Regression: Gradient-based solver with configurable learning rate/epochs, separate class instances for raw and PCA-transformed data.

Modular PCA: Eigen decomposition from scratch, explained variance calculation, elbow plot visualization for choosing components, and automated feature selection for thresholded variance retention.

Explicit Preprocessing Pipeline:

Missing-value imputation (median/mode)

Categorical encoding (custom mapping)

Standardization using train set mean/std

Modular train-test splitting—random permutation, adjustable test size

Diagnostics and Visualization:

Confusion matrices for all model/feature regimes

Explained variance print/plot

Accuracy metrics for SVM, logistic regression (raw), and logistic regression (PCA)

Elbow method for PCA, comparison of model performance


Implementation Highlights
All models (SVM, Logistic Regression, and PCA) coded from first principles.

Separate training and evaluation for raw and PCA-reduced feature sets.

Integrated variance analysis and visualization of principal components.

Systematic comparison of classic and dimensionality-reduced regimes for both model types.

Designed for transparent research diagnostics and reproducibility—ideal for education, benchmarking, or as a template for further extension.

Results
SVM Accuracy (no PCA): ~0.78

Logistic Regression Accuracy (no PCA): ~0.79

Logistic Regression Accuracy (with PCA): ~0.61

Confusion matrices, explained variance ratios, and performance plots provided in the notebook.
