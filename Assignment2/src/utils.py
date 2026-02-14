import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_models():
    """Load all models from the all_models.pkl file"""
    try:
        # Get the directory where this utils.py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to Assignment2 folder
        base_dir = os.path.dirname(current_dir)
        # Build path to models file
        models_path = os.path.join(base_dir, 'models', 'all_models.pkl')
        
        # Try absolute path first
        if os.path.exists(models_path):
            with open(models_path, 'rb') as f:
                data = pickle.load(f)
            return data
        
        # Fallback to relative path
        relative_path = 'models/all_models.pkl'
        if os.path.exists(relative_path):
            with open(relative_path, 'rb') as f:
                data = pickle.load(f)
            return data
            
        print(f"Models not found at: {models_path} or {relative_path}")
        return None
        
    except FileNotFoundError as e:
        print(f"FileNotFoundError: Models not found. {e}")
        return None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def preprocess_uploaded_data(df, scaler, feature_names):
    from sklearn.preprocessing import LabelEncoder
    
    # Remove target column if present
    if 'y' in df.columns:
        y_true = df['y'].copy()
        df = df.drop('y', axis=1)
        # Encode target
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
    else:
        y_true = None
    
    # Ensure all required features are present
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Select only the required features in correct order
    df = df[feature_names]
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    return X_scaled, y_true

def predict_with_model(model, X):
    predictions = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return predictions, probabilities
    else:
        return predictions, None

def calculate_metrics(y_true, y_pred, y_proba=None):
    from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                                recall_score, f1_score, matthews_corrcoef,
                                confusion_matrix, classification_report)
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics, cm, cr
