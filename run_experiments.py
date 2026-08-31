import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Sử dụng relative import sạch, không cần sys.path hack nếu có thể,
# nhưng vì chạy từ thư mục gốc và thư mục con là src, ta import bình thường:
from src.preprocessing import get_preprocessing_pipeline
from src.models.logistic_regression import get_logistic_regression
from src.models.perceptron import get_perceptron
from src.models.knn_classifier import get_knn_classifier
from src.metrics import calculate_metrics

def format_params(params):
    if not isinstance(params, dict):
        return str(params)
    return str({k: v.item() if hasattr(v, 'item') else v for k, v in params.items()})

def main():
    os.makedirs('experiments/figures', exist_ok=True)
    results_file = 'experiments/results.csv'
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Best_Parameters', 'CV_Score', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC'])
    
    # 1. Load dataset
    df = pd.read_csv('data/heart_cleveland_upload.csv')
    
    # Tạo EDA charts
    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='condition')
    plt.title('Target Distribution')
    plt.savefig('experiments/figures/target_distribution.png')
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.savefig('experiments/figures/correlation_matrix.png')
    plt.close()
    
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Build preprocessing
    preprocessor = get_preprocessing_pipeline()
    
    # Function helper cho evaluation
    def evaluate_model(name, pipeline, param_grid):
        print(f"Running {name}...")
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        
        # Nếu model có predict_proba hoặc decision_function
        y_score = None
        if hasattr(grid, "predict_proba"):
            y_score = grid.predict_proba(X_test)[:, 1]
        elif hasattr(grid, "decision_function"):
            y_score = grid.decision_function(X_test)
            
        metrics = calculate_metrics(y_test, y_pred, y_score)
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, format_params(grid.best_params_), grid.best_score_, 
                             metrics['Accuracy'], metrics['Precision'], metrics['Recall'], 
                             metrics['F1_Score'], metrics['ROC_AUC']])
        return grid

    # 4. Perceptron + GridSearchCV
    perc_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', get_perceptron())
    ])
    perc_param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1],
        'model__penalty': [None, 'l2', 'l1']
    }
    evaluate_model('Perceptron', perc_pipeline, perc_param_grid)
    
    # 5. Logistic Regression + GridSearchCV
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', get_logistic_regression())
    ])
    lr_param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs', 'liblinear']
    }
    evaluate_model('Logistic Regression', lr_pipeline, lr_param_grid)
    
    # 6. KNN + GridSearchCV
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', get_knn_classifier())
    ])
    knn_param_grid = {
        'model__n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance']
    }
    knn_grid = evaluate_model('KNN Classifier', knn_pipeline, knn_param_grid)
    
    # Vẽ biểu đồ K vs CV Score (lấy từ KNN grid)
    results_knn = pd.DataFrame(knn_grid.cv_results_)
    plt.figure(figsize=(8, 5))
    for weight in ['uniform', 'distance']:
        subset = results_knn[results_knn['param_model__weights'] == weight]
        plt.plot(subset['param_model__n_neighbors'], subset['mean_test_score'], marker='o', label=f'weights={weight}')
    plt.title('KNN: K vs Validation Accuracy')
    plt.xlabel('n_neighbors (K)')
    plt.ylabel('CV Accuracy')
    plt.legend()
    plt.savefig('experiments/figures/knn_k_vs_score.png')
    plt.close()
    
    # 7. Phân tích PCA (Vẽ biểu đồ explained variance)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    pca_temp = PCA().fit(X_train_preprocessed)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_temp.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.title('PCA Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.savefig('experiments/figures/pca_explained_variance.png')
    plt.close()

    # 8. PCA + Logistic Regression
    pca_lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('model', get_logistic_regression())
    ])
    pca_lr_param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100]
    }
    evaluate_model('PCA + Logistic Regression', pca_lr_pipeline, pca_lr_param_grid)
    
    # 9. PCA + KNN
    pca_knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('model', get_knn_classifier())
    ])
    pca_knn_param_grid = {
        'model__n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance']
    }
    evaluate_model('PCA + KNN Classifier', pca_knn_pipeline, pca_knn_param_grid)

    # 10. Generate model comparison figure
    print("Generating Model Comparison Chart...")
    results_df = pd.read_csv(results_file)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test_Accuracy', y='Model', data=results_df.sort_values('Test_Accuracy', ascending=False), hue='Model', dodge=False)
    plt.title('Model Comparison - Test Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('experiments/figures/model_comparison.png')
    plt.close()
    
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
