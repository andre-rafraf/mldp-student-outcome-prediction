# Student Outcome Prediction

A machine learning application that predicts whether a student is likely to:

- Dropout
- Remain Enrolled
- Graduate

The prediction is based on demographic, financial, admission and Semester 1 academic information.

## Business Problem

Educational institutions may benefit from identifying students who are at risk of dropping out.

This project provides an early decision-support tool that can help institutions identify students who may require additional academic or financial support.

The prediction should not be treated as a final decision about a student.

## Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

The dataset contains:

- 4,424 student records
- Demographic information
- Admission information
- Financial information
- Semester academic results
- Three target classes: Dropout, Enrolled and Graduate

## Machine Learning Task

This is a multiclass classification problem.

The model predicts one of three outcomes:

1. Dropout
2. Enrolled
3. Graduate

## Data Preparation

The main preparation steps include:

- Checking missing values and duplicate records
- Removing Semester 2 features to prevent data leakage
- Separating numerical and categorical variables
- Scaling numerical variables
- One-hot encoding categorical variables
- Using a stratified train-test split

## Feature Engineering

Two Semester 1 features were created:

### Semester 1 Pass Rate

```text
Approved units / Enrolled units
```

### Semester 1 Completion Gap

```text
Enrolled units - Approved units
```

These features represent student progress during Semester 1.

## Models Compared

The following models were evaluated:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier

Logistic Regression was selected as the final model because it achieved the strongest Macro F1 score while remaining interpretable.

## Final Model Performance

| Metric | Test Result |
|---|---:|
| Accuracy | 0.715 |
| Macro F1 | 0.678 |
| Dropout Recall | 0.673 |

The final model uses Logistic Regression with tuned hyperparameters.

## Streamlit Application

The Streamlit application allows users to:

- Enter student information
- Enter Semester 1 academic results
- Validate invalid combinations of inputs
- Generate a predicted student outcome
- View probabilities for all three possible outcomes

## Project Files

```text
Student Outcome/
├── Student Outcome.ipynb
├── app.py
├── data.csv
├── student_outcome_model.joblib
├── requirements.txt
└── README.md
```

## Running the Application Locally

Activate the project environment:

```powershell
conda activate mldp
```

Move into the project folder:

```powershell
cd "C:\Users\andre\Downloads\Student Outcome"
```

Start the Streamlit application:

```powershell
streamlit run app.py
```

Alternatively:

```powershell
python -m streamlit run app.py
```

The application should open at:

```text
http://localhost:8502
```

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Matplotlib
- joblib
- Jupyter Notebook

## Repository

GitHub repository:

```text
https://github.com/andre-rafraf/mldp-student-outcome-prediction
```

## Disclaimer

This application is an educational machine learning project.

Its predictions should be used only as an early support indicator and should not be used as the sole basis for academic decisions.