import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef,confusion_matrix, classification_report)
from data_preprocessing import load_and_preprocess_data
import warnings
warnings.filterwarnings('ignore')


# Evaluate a model and return all required metrics

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    # Get probability predictions for AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred, average='binary'),
        'Recall': recall_score(y_test, y_pred, average='binary'),
        'F1': f1_score(y_test, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"AUC Score: {metrics['AUC']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1']:.4f}")
    print(f"MCC Score: {metrics['MCC']:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    return metrics, cm, cr


# Train all required models and save them

def train_all_models():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=36),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=36, max_depth=8),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=36, max_depth=5, 
                                learning_rate=0.1, eval_metric='logloss')
    }
    
    # Train and evaluate all models
    all_metrics = []
    trained_models = {}
    confusion_matrices = {}
    classification_reports = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        metrics, cm, cr = evaluate_model(model, X_test, y_test, model_name)
        all_metrics.append(metrics)
        trained_models[model_name] = model
        confusion_matrices[model_name] = cm
        classification_reports[model_name] = cr
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(metrics_df.to_string(index=False))
    
    # Create models directory if it doesn't exist
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        models_dir = os.path.join('..', 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, 'all_models.pkl')
    print(f"Saving to: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models': trained_models,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics_df,
            'confusion_matrices': confusion_matrices,
            'classification_reports': classification_reports
        }, f)
    
    return metrics_df, trained_models, scaler, feature_names

if __name__ == "__main__":
    metrics_df, models, scaler, feature_names = train_all_models()