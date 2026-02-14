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
                             matthews_corrcoef, confusion_matrix, classification_report)
from data_preprocessing import load_and_preprocess_data
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages for cleaner output


def evaluate_model(model, X_test, y_test, model_name):
    
    # ==========================================
    # STEP 1: MAKE PREDICTIONS
    # ==========================================
    
    # Predict class labels (0 or 1) for test data
    y_pred = model.predict(X_test)
    
    # Get probability predictions if model supports it
    # Probabilities are needed for AUC calculation
    if hasattr(model, 'predict_proba'):
        # Get probability of positive class (class 1)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Some models don't support probabilities, use predictions instead
        y_proba = y_pred
    
    # ==========================================
    # STEP 2: CALCULATE PERFORMANCE METRICS
    # ==========================================
    
    metrics = {
        'Model': model_name,
        
        'Accuracy': accuracy_score(y_test, y_pred),
       
        'AUC': roc_auc_score(y_test, y_proba),
        
        'Precision': precision_score(y_test, y_pred, average='binary'),
        
        'Recall': recall_score(y_test, y_pred, average='binary'),
        
        'F1': f1_score(y_test, y_pred, average='binary'),
        
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    # ==========================================
    # STEP 3: GENERATE CONFUSION MATRIX
    # ==========================================
    
    cm = confusion_matrix(y_test, y_pred)
    
    # ==========================================
    # STEP 4: GENERATE CLASSIFICATION REPORT
    # ==========================================
    
    # Detailed report with precision, recall, f1 for each class
    cr = classification_report(y_test, y_pred)
    
    # ==========================================
    # STEP 5: DISPLAY RESULTS
    # ==========================================
    
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f} (Overall correctness)")
    print(f"AUC Score: {metrics['AUC']:.4f} (Discrimination ability)")
    print(f"Precision: {metrics['Precision']:.4f} (Positive prediction accuracy)")
    print(f"Recall:    {metrics['Recall']:.4f} (True positive detection rate)")
    print(f"F1 Score:  {metrics['F1']:.4f} (Precision-Recall balance)")
    print(f"MCC Score: {metrics['MCC']:.4f} (Correlation coefficient)")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           No       Yes")
    print(f"Actual No  {cm[0,0]:4d}     {cm[0,1]:4d}  (TN, FP)")
    print(f"      Yes  {cm[1,0]:4d}     {cm[1,1]:4d}  (FN, TP)")
    
    return metrics, cm, cr


def train_all_models():
    
    # ==========================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ==========================================
    
    print("\n" + "🚀 " * 25)
    print("STARTING MODEL TRAINING PIPELINE")
    print("🚀 " * 25 + "\n")
    
    print("📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Number of features: {len(feature_names)}")
    
    # ==========================================
    # STEP 2: INITIALIZE ALL MODELS
    # ==========================================
    
    print("\n🤖 Initializing machine learning models...")
    
    models = {
        # MODEL 1: Logistic Regression
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=36),
        
        # MODEL 2: Decision Tree
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        
        # MODEL 3: K-Nearest Neighbors (KNN)
        'KNN': KNeighborsClassifier(n_neighbors=3),
        
        # MODEL 4: Naive Bayes
        'Naive Bayes': GaussianNB(),
        
        # MODEL 5: Random Forest (Ensemble)
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=36, max_depth=8),
        
        # MODEL 6: XGBoost (Ensemble)
        'XGBoost': XGBClassifier(n_estimators=100, random_state=36, max_depth=5, learning_rate=0.1, eval_metric='logloss')
    }
    
    print(f"✓ Initialized {len(models)} models:")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i}. {name}")
    
    # ==========================================
    # STEP 3: TRAIN AND EVALUATE EACH MODEL
    # ==========================================
    
    print("\n" + "🎯 " * 25)
    print("TRAINING AND EVALUATING MODELS")
    print("🎯 " * 25)
    
    # Storage for results
    all_metrics = []                  
    trained_models = {}               
    confusion_matrices = {}           
    classification_reports = {}      
    
    # Train each model one by one
    for model_name, model in models.items():
        print(f"\n{'─'*60}")
        print(f"🔄 Training: {model_name}")
        print(f"{'─'*60}")
        
        model.fit(X_train, y_train)
        print(f"✓ Training completed")
        
        metrics, cm, cr = evaluate_model(model, X_test, y_test, model_name)
        
        # Store results
        all_metrics.append(metrics)
        trained_models[model_name] = model
        confusion_matrices[model_name] = cm
        classification_reports[model_name] = cr
    
    # ==========================================
    # STEP 4: CREATE COMPARISON TABLE
    # ==========================================
    
    # Convert metrics list to DataFrame for easy comparison
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON TABLE")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80)
    
    # Find best model for each metric
    print("\n🏆 BEST PERFORMERS:")
    print(f"  Highest Accuracy:  {metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']} "
          f"({metrics_df['Accuracy'].max():.4f})")
    print(f"  Highest AUC:       {metrics_df.loc[metrics_df['AUC'].idxmax(), 'Model']} "
          f"({metrics_df['AUC'].max():.4f})")
    print(f"  Highest F1:        {metrics_df.loc[metrics_df['F1'].idxmax(), 'Model']} "
          f"({metrics_df['F1'].max():.4f})")
    
    # ==========================================
    # STEP 5: SAVE EVERYTHING TO DISK
    # ==========================================
    
    print("\n💾 Saving models and results...")
    
    # Determine correct path for models directory
    # Try current directory first, then parent directory
    models_dir = 'models'
    if not os.path.exists(models_dir):
        models_dir = os.path.join('..', 'models')
    
    # Create directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"  ✓ Created directory: {models_dir}")
    
    # Save everything in one pickle file for easy loading
    model_path = os.path.join(models_dir, 'all_models.pkl')
    print(f"  ✓ Saving to: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models': trained_models,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics_df,
            'confusion_matrices': confusion_matrices,
            'classification_reports': classification_reports
        }, f)
    
    print(f"  ✓ Saved successfully!")
    
    # ==========================================
    # STEP 6: FINAL SUMMARY
    # ==========================================
    
    print("\n" + "✨ " * 25)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("✨ " * 25)
    print(f"\n📦 Package contents saved to: {model_path}")
    print(f"  • {len(trained_models)} trained models")
    print(f"  • Scaler fitted on {len(X_train)} training samples")
    print(f"  • {len(feature_names)} feature names")
    print(f"  • Performance metrics for all models")
    print(f"  • Confusion matrices and classification reports")
    print("\n🎉 Models are ready for deployment in Streamlit app!\n")
    
    return metrics_df, trained_models, scaler, feature_names


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    """
    This block runs when the script is executed directly.
    It trains all models and saves them for later use.
    """
    
    # Run the complete training pipeline
    metrics_df, models, scaler, feature_names = train_all_models()
    
    # Display final message
    print("=" * 80)
    print("You can now use these models in your Streamlit app!")
    print("The app will load all_models.pkl and use it for predictions.")
    print("=" * 80)