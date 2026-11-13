# Diabetes Prediction — Machine Learning Learning Project

An educational classification project using the Pima Indians Diabetes dataset. The script demonstrates data preparation, visualization, Random Forest classification, and evaluation.

> **Medical disclaimer:** This is a learning exercise, not a medical device or diagnostic tool. It must not be used to diagnose diabetes, rule it out, or guide treatment.

## Current workflow

- Load the `diabetes.csv` dataset.
- Clean selected values and visualize feature distributions and correlations.
- Split the data into training and test sets.
- Train a Random Forest classifier.
- Display accuracy, a confusion matrix, and a classification report.

## Technology

Python, Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Run locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Main.py
```

The dataset file is expected at the path used by the script. Check that path if running from another working directory.

## Results and limitations

The existing README reports 85.71% accuracy and this confusion matrix for one run:

```text
[[89, 11],
 [18, 36]]
```

These figures are dataset- and split-specific, not a guarantee of performance on other populations or clinical settings. Accuracy alone is insufficient for a health-related classifier. Report sensitivity/recall, specificity, precision, F1, ROC-AUC, calibration, and confidence intervals where possible.

The dataset has clinically meaningful missing-value conventions. Replacing every zero with a mean can distort valid measurements and should not be done indiscriminately. Document which fields use zero as a missing-value code and fit imputers only on training data.

## Recommended improvements

- Use a reproducible split and scikit-learn Pipeline.
- Evaluate class imbalance and threshold trade-offs.
- Add tests and document dataset provenance and license.
- Clearly communicate limitations and prohibit clinical use.

## Author

Piyush Rai — [GitHub](https://github.com/Piyushupc)
