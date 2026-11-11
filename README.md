An end-to-end machine learning investigation of industrial equipment failure prediction using Python, statistical analysis, and classification algorithms.

Project Overview:

This project investigates whether operational sensor measurements can be used to predict industrial equipment failure.

Using the AI4I 2020 Predictive Maintenance Dataset, I developed and evaluated Logistic Regression and Random Forest classifiers, investigated relationships between operating variables, and optimized a classification threshold to improve failure detection.

The project demonstrates data exploration, statistical reasoning, preprocessing, model evaluation, and interpretation.

Dataset:

Source: UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset

The dataset contains 10,000 synthetic observations representing industrial machinery operating conditions.

Predictive features:

Air temperature (K)

Process temperature (K)

Rotational speed (rpm)

Torque (Nm)

Tool wear (min)

Machine type

Target variable: Machine failure (binary classification).

The dataset has a failure rate of 3.39%, creating a class imbalance that influences model evaluation.

Methodology:

1. Exploratory Data Analysis

Assessed missing values and duplicates.

Investigated class imbalance.

Compared sensor measurements between failed and non-failed observations.

Examined feature distributions using boxplots.

Calculated Pearson correlations.

Investigated the relationship between torque and rotational speed.

2. Data Preparation

Excluded identifiers and failure-type indicators to prevent direct target leakage.

Used an 80/20 stratified training and testing split.

Applied StandardScaler to numerical variables.

Applied OneHotEncoder to machine type.

Integrated preprocessing into scikit-learn pipelines.

3. Model Development

Two classification models were evaluated using five-fold stratified cross-validation on the training dataset:

Logistic Regression

Random Forest (200 decision trees)

Models were compared using accuracy, precision, recall, F1-score, and Average Precision.

4. Threshold Optimization

The Random Forest classification threshold was adjusted from 0.50 to approximately 0.305 by maximizing F1-score using out-of-fold training predictions.

This improved failure detection at the cost of additional false-positive alerts.

The held-out test dataset was not used to select the threshold.

Results:

Cross-Validation Model Comparison

Metric:

Logistic Regression

Random Forest

Accuracy

97.0%

98.0%

Precision

76.1%

87.4%

Recall

19.9%

48.0%

F1-score

0.312

0.618

Average Precision

0.469

0.770

Mean five-fold cross-validation results using the default classification threshold.

Final Random Forest Test Performance

Metric

Result

Accuracy

98.2%

Precision

72.5%

Recall

73.5%

F1-score

0.730

Average Precision

0.789

Selected threshold

0.305

The final classifier correctly identified 50 of the 68 failures in the held-out test set, while missing 18 failures.

Feature Importance

Permutation importance analysis on a separate training-derived validation subset identified the following ranking:

Torque

Rotational speed

Air temperature

Tool wear

Process temperature

Machine type

These results suggest that combinations of operating conditions contribute to failure prediction. Feature importance does not establish causation.

Technologies

Python

pandas and NumPy

Matplotlib

scikit-learn

Google Colab

Git and GitHub

Project Notebooks

Exploratory Data Analysis

Data quality assessment, descriptive statistics, failure distribution, correlation analysis, and visualizations.

Model Development and Evaluation

Preprocessing, Logistic Regression, Random Forest, cross-validation, threshold optimization, final test evaluation, and feature importance.

Limitations

The AI4I dataset is synthetic and does not represent real industrial sensor measurements.

The final model missed approximately 26.5% of failures.

The random train-test split does not establish performance on future observations or entirely new machines.

The threshold was selected using pooled out-of-fold predictions, which may produce optimistic training estimates.

Correlated features complicate the interpretation of individual feature importance scores.

Future Work

Potential extensions include cost-sensitive learning, probability calibration, more rigorous threshold selection, and evaluation using real industrial time-series sensor data.

Reproducibility

The analysis was developed in Google Colab.

To reproduce the project:

Download ai4i2020.csv from the UCI dataset page linked above.

Open the notebooks in Google Colab.

Upload ai4i2020.csv to the Colab runtime.

Run the notebook cells sequentially.

The experiments use fixed random seeds to support reproducibility. Exact results may depend on library versions.

Project Context

Independent Python and machine learning portfolio project developed to strengthen practical experience in data science, statistical analysis, and predictive modelling.
