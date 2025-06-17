# Loan Defaulters Prediction System

## Overview

This project presents a **Loan Defaulter Prediction System** developed to assess the likelihood of a customer defaulting on a loan, based on historical banking data. Using state-of-the-art machine learning techniques, the system predicts loan risk with a high degree of accuracy and is deployed using Flask for real-time predictions.


## Objective

To build a robust machine learning model that predicts whether a customer will default on a loan using multiple business and financial indicators, and to deploy the model via a web interface.


## Dataset Summary

- **Source**: Internal banking dataset  
- **Observations**: 150,000 rows  
- **Features**: 26 input variables including loan term, employment numbers, disbursement amount, franchise status, urban/rural category, SBA approval values, etc.  
- **Target**: `MIS_Status` – Indicates whether the customer is a **defaulter (CHGOFF)** or **non-defaulter (PIF)**


## Tools & Technologies Used

- **Languages**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, Flask, Pickle
- **Web Framework**: Flask
- **Model Deployment**: HTML + Flask API
- **Data Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Accuracy, Confusion Matrix, ROC-AUC


## Key Features

- Extensive data cleaning, imputation, and feature engineering
- Feature selection using **ExtraTreesClassifier** and **XGBoost**
- Model comparison using:
  - **XGBoost** (Accuracy: ~99%)
  - **Random Forest**
  - **Naive Bayes**
  - **LightGBM**
- Deployed web app allows real-time loan default prediction with a clean UI


## Project Structure

```
Loan-Defaulters-Prediction/
├── app.py                    # Main Flask backend for deployment
├── Loan_final.py            # Model training and evaluation script
├── Model.pkl                # Serialized trained model
├── index.html               # Frontend user interface
├── style.css                # CSS styling for the UI
├── Requirement Project document.docx  # Project requirements documentation
├── Data dictionary.docx     # Description of all dataset variables
└── README.md                # Project documentation
```


## Deployment

The web application is hosted using Flask with an HTML interface. The model was serialized using `pickle` and loaded into the Flask app for prediction.


### How to Run

1. Clone this repo
2. Install dependencies using `pip install -r requirements.txt`
3. Run `python app.py`
4. Access the app via `localhost:5000`


## Usage

The user inputs loan-related details such as:
- Loan Term
- Number of Employees
- Urban/Rural Indicator
- SBA Approval amount, etc.

The application predicts whether the loan is **likely to default** or **safe to approve**.


## Impact

This system empowers banks and financial institutions to:
- Reduce loan default risk
- Speed up approval workflows
- Maintain financial health through predictive insights


## Author

**Surya Vamsi Patiballa**  
MS in Data Science at George Washington University

- Email  :-  svamsi2002@gmail.com
- LinkedIn  :-  https://www.linkedin.com/in/surya-patiballa-b724851aa/
- Resume  :-  https://drive.google.com/file/d/178IYcArC6YYVdJiIwRmJYodzKZ-JXe-D/view?usp=sharing


> “Smart lending starts with smart predictions.”
