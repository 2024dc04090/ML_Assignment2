import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# Load and preprocess the Bank Marketing dataset

def load_and_preprocess_data(filepath='data/bank.csv'):
    if not os.path.exists(filepath):
        filepath = os.path.join('..', filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found!")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Display basic info
    print("Dataset shape:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    
    target_col = None
    possible_targets = ['y', 'target', 'deposit', 'label', 'class']
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Target column not found! Available columns: {df.columns.tolist()}")
    
    print(f"\nUsing '{target_col}' as target column")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode target variable (yes/no to 1/0)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Handle missing values in numerical columns
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    print("\nData preprocessing completed successfully!")