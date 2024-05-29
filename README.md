Heart Disease Prediction Using Machine Learning

Overview
This repository contains a project on predicting heart disease using various machine-learning algorithms. The goal is to build a predictive model that can accurately classify whether a person has heart disease based on a set of medical attributes.

Table of Contents
Introduction
Dataset
Model Training
Evaluation
Results


Introduction
Heart disease is one of the leading causes of death globally. Early detection and treatment can significantly improve survival rates. This project aims to utilize machine learning techniques to predict the presence of heart disease, aiding healthcare professionals in making informed decisions.

Dataset
The dataset used in this project is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. It contains 303 instances with 14 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate achieved, exercise-induced angina, oldpeak, slope, number of major vessels, and thal.


Preprocess the data: Clean and preprocess the dataset to prepare it for training.
Train the model: Use various machine learning algorithms to train the model.
Evaluate the model: Assess the performance of the model using appropriate metrics.
Make predictions: Use the trained model to make predictions on new data.
The main script to run the project is heart_disease_prediction.py. You can execute it as follows:

Model Training
The following machine-learning algorithms are implemented in this project:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gradient Boosting
The training process involves splitting the dataset into training and testing sets, training the model on the training set, and evaluating it on the testing set.

Evaluation
The models are evaluated using the following metrics:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
These metrics provide a comprehensive view of the model's performance.

Results
The results of the model training and evaluation are documented in the results directory. This includes performance metrics and visualizations such as ROC curves.

