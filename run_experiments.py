import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("src")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from preprocessing import get_preprocessing_pipeline
from models.logistic_regression import get_logistic_regression
from models.perceptron import get_perceptron
from models.knn_classifier import get_knn_classifier
from metrics import calculate_metrics

def format_params(params):
    if not isinstance(params, dict):
        return str(params)
    return str({k: v.item() if hasattr(v, 'item') else v for k, v in params.items()})

def main():
    os.makedirs('experiments/figures', exist_ok=True)
    results_file = 'experiments/results.csv'
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Hyperparameters', 'CV_Accuracy', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1'])
    
    # Load dataset
    df = pd.read_csv('data/heart_cleveland_upload.csv')
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Logistic Regression
    print("Running Logistic Regression...")
    lr_pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('model', get_logistic_regression())
    ])
    lr_param_grid = {
        'model__C': [0.1, 1.0, 10.0]
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_pred = lr_grid.predict(X_test)
    lr_metrics = calculate_metrics(y_test, lr_pred)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Logistic Regression', format_params(lr_grid.best_params_), lr_grid.best_score_, lr_metrics['Accuracy'], lr_metrics['Precision'], lr_metrics['Recall'], lr_metrics['F1_Score']])

    # 2. Perceptron
    print("Running Perceptron...")
    perc_pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('model', get_perceptron())
    ])
    perc_param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__penalty': [None, 'l2', 'l1']
    }
    perc_grid = GridSearchCV(perc_pipeline, perc_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    perc_grid.fit(X_train, y_train)
    perc_pred = perc_grid.predict(X_test)
    perc_metrics = calculate_metrics(y_test, perc_pred)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Perceptron', format_params(perc_grid.best_params_), perc_grid.best_score_, perc_metrics['Accuracy'], perc_metrics['Precision'], perc_metrics['Recall'], perc_metrics['F1_Score']])

    # 3. KNN Classifier
    print("Running KNN Classifier...")
    knn_pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('model', get_knn_classifier())
    ])
    knn_param_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11],
        'model__weights': ['uniform', 'distance']
    }
    knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    knn_pred = knn_grid.predict(X_test)
    knn_metrics = calculate_metrics(y_test, knn_pred)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['KNN Classifier', format_params(knn_grid.best_params_), knn_grid.best_score_, knn_metrics['Accuracy'], knn_metrics['Precision'], knn_metrics['Recall'], knn_metrics['F1_Score']])

    # 4. PCA + Logistic Regression
    print("Running PCA + Logistic Regression...")
    pca_lr_pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('pca', PCA()),
        ('model', get_logistic_regression())
    ])
    pca_param_grid = {
        'pca__n_components': [0.8, 0.9, 0.95], # keep 80%, 90% or 95% of variance
        'model__C': [0.1, 1.0, 10.0]
    }
    pca_grid = GridSearchCV(pca_lr_pipeline, pca_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    pca_grid.fit(X_train, y_train)
    pca_pred = pca_grid.predict(X_test)
    pca_metrics = calculate_metrics(y_test, pca_pred)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PCA + Logistic Regression', format_params(pca_grid.best_params_), pca_grid.best_score_, pca_metrics['Accuracy'], pca_metrics['Precision'], pca_metrics['Recall'], pca_metrics['F1_Score']])

    # 5. Model Comparison Chart
    print("Generating Model Comparison Chart...")
    results_df = pd.read_csv(results_file)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test_Accuracy', y='Model', data=results_df.sort_values('Test_Accuracy', ascending=False), hue='Model')
    plt.title('Model Comparison - Test Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('experiments/figures/model_comparison_accuracy.png')
    
    print("All experiments completed successfully.")

if __name__ == '__main__':
    main()
