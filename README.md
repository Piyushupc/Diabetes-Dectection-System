# Diabetes Prediction Using Machine Learning

A small educational machine-learning project that explores binary classification using the Pima Indians Diabetes Dataset and a Random Forest classifier.

> **Medical disclaimer:** This project is for learning and experimentation only. It is not a medical device, diagnostic tool, or substitute for advice from a qualified healthcare professional. Do not use its predictions to make healthcare decisions.

## Features

- Load the `diabetes.csv` dataset.
- Perform basic data cleaning and exploratory visualization.
- Train a Random Forest classifier.
- Evaluate predictions using accuracy, a confusion matrix, and a classification report.

## Technology stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn

## Project structure

```text
Main.py
diabetes.csv
README.md
```

## Run locally

Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

Run:

```bash
python Main.py
```

## Evaluation

The existing project documentation reports an accuracy of 85.71% and the following confusion matrix:

```text
[[89, 11],
 [18, 36]]
```

These are previously documented results, not independently reproduced in this documentation update. Re-run the project and record the exact split, random seed, preprocessing, and class-wise metrics before using the figure in a portfolio or report.

## Modeling considerations

- Zero values in medical measurements can mean different things; do not automatically replace every zero without checking the feature definition.
- Fit imputers and other preprocessing only on the training split.
- Use stratified splitting and report precision, recall, F1, ROC-AUC, and a confusion matrix.
- Check class imbalance and calibration.
- The dataset has population and sampling limitations; results may not generalize to other populations.

## Recommended improvements

- Use a Pipeline and ColumnTransformer.
- Add cross-validation and tests.
- Document dataset provenance and license.
- Add a reproducible training/evaluation script.
- Keep the medical-use disclaimer visible.

## Author

Piyush Rai — [GitHub](https://github.com/Piyushupc)
