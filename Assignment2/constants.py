"""
Constants module for Streamlit ML Application
Contains all constants, enums, and configuration data
"""

# Application configuration
APP_CONFIG = {
    'title': 'ML Assignment 2',
    'layout': 'wide'
}

# Model descriptions and analysis
MODEL_DESCRIPTIONS = {
    'Logistic Regression': {
        'desc': 'A linear model for binary classification using the logistic function.',
        'strengths': 'Fast training, interpretable coefficients, works well with linearly separable data',
        'weaknesses': 'Limited to linear decision boundaries, may underfit complex patterns',
        'use_case': 'Best for baseline modeling and when interpretability is crucial'
    },
    'Decision Tree': {
        'desc': 'A tree-based model that makes decisions by splitting data based on feature values.',
        'strengths': 'Highly interpretable, handles non-linear relationships, no feature scaling needed',
        'weaknesses': 'Prone to overfitting, unstable to small data changes',
        'use_case': 'Good for understanding feature importance and decision paths'
    },
    'KNN': {
        'desc': 'Instance-based learning that classifies based on k-nearest neighbors.',
        'strengths': 'Simple concept, no training phase, effective for small datasets',
        'weaknesses': 'Computationally expensive, sensitive to feature scaling and noise',
        'use_case': 'Suitable when similar instances should have similar predictions'
    },
    'Naive Bayes': {
        'desc': 'Probabilistic classifier based on Bayes theorem with independence assumptions.',
        'strengths': 'Very fast, works well with high-dimensional data, requires small training data',
        'weaknesses': 'Strong independence assumption may not hold in reality',
        'use_case': 'Excellent for text classification and when speed is priority'
    },
    'Random Forest': {
        'desc': 'Ensemble of decision trees using bagging to reduce overfitting.',
        'strengths': 'Robust to overfitting, handles missing values, provides feature importance',
        'weaknesses': 'Less interpretable, can be slow with many trees',
        'use_case': 'Great balance of accuracy and robustness for most problems'
    },
    'XGBoost': {
        'desc': 'Advanced gradient boosting algorithm that builds trees sequentially.',
        'strengths': 'State-of-the-art performance, handles imbalanced data, built-in regularization',
        'weaknesses': 'Requires careful hyperparameter tuning, can overfit if not tuned',
        'use_case': 'Top choice for competitions and production systems requiring best accuracy'
    }
}

# Metric information with icons
METRIC_INFO = [
    ("Accuracy"),
    ("AUC"),
    ("Precision"),
    ("Recall"),
    ("F1"),
    ("MCC")
]

# Metrics to highlight in table
METRICS_TO_HIGHLIGHT = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

# Metrics format configuration
METRIC_FORMAT = {
    'Accuracy': '{:.4f}',
    'AUC': '{:.4f}',
    'Precision': '{:.4f}',
    'Recall': '{:.4f}',
    'F1': '{:.4f}',
    'MCC': '{:.4f}'
}

# Chart configuration
CHART_CONFIG = {
    'accuracy_comparison': {
        'title': 'Model Accuracy Comparison',
        'ylabel': 'Accuracy Score',
        'xlabel': 'Model',
        'figsize': (10, 6)
    },
    'multi_metric': {
        'title': 'Multi-Metric Performance Comparison',
        'ylabel': 'Score',
        'xlabel': 'Model',
        'figsize': (10, 6),
        'metrics': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'ylim': [0.7, 0.9]
    },
    'auc_comparison': {
        'title': 'AUC Score Comparison (Higher is Better)',
        'xlabel': 'AUC Score',
        'figsize': (12, 5),
        'xlim': [0.75, 1.0]
    },
    'confusion_matrix': {
        'figsize': (6, 5),
        'cmap': 'RdYlGn',
        'labels': ['No', 'Yes']
    }
}

# Analysis text
ANALYSIS_TEXT = {
    'accuracy': """<small><strong>Analysis:</strong> This chart compares the overall accuracy of all models. 
        Higher bars indicate better performance. XGBoost and Random Forest show superior accuracy, 
        while Naive Bayes has the lowest accuracy among all models.</small>""",
    
    'multi_metric': """<small><strong>Analysis:</strong> This grouped bar chart shows performance across multiple metrics. 
        XGBoost (purple) consistently ranks high across all metrics, demonstrating well-rounded performance. 
        Precision and Recall balance varies across models, with Naive Bayes showing higher recall but lower precision.</small>""",
    
    'auc': """<small><strong>Analysis:</strong> AUC (Area Under ROC Curve) measures the model's ability to distinguish 
        between classes. Scores closer to 1.0 are better. XGBoost (0.924) and Random Forest (0.913) show excellent 
        discriminative power, significantly outperforming other models. All models achieve AUC > 0.80, indicating 
        good classification capability.</small>"""
}

# UI Messages
UI_MESSAGES = {
    'models_not_found': "Models not found! Please train the models first by running model_training.py",
    'models_not_found_hint': "Run the following command in your terminal: `python src/model_training.py`",
    'file_upload_success': "<strong>File uploaded successfully!</strong>",
    'processing': "Processing your data...",
    'prediction_error': "Error during prediction: {}",
    'file_load_error': "Error loading file: {}",
    'ensure_features': "Please ensure your CSV has the same features as the training data.",
}

# Test data requirements
TEST_DATA_REQUIREMENTS = {
    'title': 'Expected CSV Format',
    'description': 'Your CSV should contain these columns:',
    'optional_note': '<strong>Optional:</strong> Include \'deposit\' or \'y\' column for ground truth labels to calculate evaluation metrics.'
}

# File upload info
FILE_UPLOAD_INFO = """<strong>Requirements:</strong> Your CSV should contain the same features as training data. 
    Optionally include a <code>deposit</code> or <code>y</code> column for ground truth to calculate metrics."""

# AUC thresholds
AUC_THRESHOLDS = {
    'excellent': 0.90,
    'good': 0.85
}

# Tab names
TAB_NAMES = {
    'training': 'Training Results',
    'testing': 'Model Testing'
}

# Section headers
SECTION_HEADERS = {
    'performance_overview': 'Model Performance Comparison',
    'metrics_table': 'Detailed Metrics Table',
    'model_analysis': 'Detailed Model Analysis',
    'performance_viz': 'Performance Visualizations',
    'test_models': 'Test Models on Your Data',
    'upload_data': 'Upload Test Data',
    'prediction_results': 'Prediction Results',
    'model_evaluation': 'Model Evaluation'
}

# Possible target column names
POSSIBLE_TARGET_COLUMNS = ['y', 'target', 'deposit', 'label', 'class']

# Prediction labels
PREDICTION_LABELS = {
    0: 'No',
    1: 'Yes'
}