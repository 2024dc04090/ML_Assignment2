import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(filepath='data/bank.csv'):
    
    # ==========================================
    # STEP 1: LOCATE AND LOAD THE DATASET
    # ==========================================
    
    # Check if file exists at given path
    if not os.path.exists(filepath):
        # Try looking in parent directory if not found
        filepath = os.path.join('..', filepath)
    
    # If still not found, raise an error
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}!")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Display basic information about the dataset
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names: {df.columns.tolist()}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("=" * 60)
    
    # ==========================================
    # STEP 2: IDENTIFY TARGET COLUMN
    # ==========================================
    
    # The target column contains what we want to predict (yes/no for deposit)
    target_col = None
    
    # List of possible names for the target column
    possible_targets = ['y', 'target', 'deposit', 'label', 'class']
    
    # Search through the DataFrame columns to find the target
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    # If no target column found, we can't proceed
    if target_col is None:
        raise ValueError(f"Target column not found! Available columns: {df.columns.tolist()}")
    
    print(f"\n✓ Using '{target_col}' as target column")
    
    # ==========================================
    # STEP 3: SEPARATE FEATURES AND TARGET
    # ==========================================
    
    # X = Features (all columns except target)
    # y = Target (what we want to predict)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # ==========================================
    # STEP 4: ENCODE TARGET VARIABLE
    # ==========================================
    
    # Convert target from text ('yes'/'no') to numbers (1/0)
    # Most ML algorithms require numerical input
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # ==========================================
    # STEP 5: IDENTIFY COLUMN TYPES
    # ==========================================
    
    # Categorical columns: Text-based features (e.g., job, education)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Numerical columns: Number-based features (e.g., age, balance)
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n📊 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"🔢 Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # ==========================================
    # STEP 6: HANDLE MISSING VALUES
    # ==========================================
    
    # Fill missing values in numerical columns with the median
    # Median is robust to outliers (unlike mean)
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
        print(f"\n✓ Imputed missing values in numerical columns using median strategy")
    
    # ==========================================
    # STEP 7: ENCODE CATEGORICAL VARIABLES
    # ==========================================
    
    # Convert categorical text to numbers
    # For example: 'admin' -> 0, 'blue-collar' -> 1, etc.
    print(f"\n🔄 Encoding categorical variables...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string first to handle any edge cases
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"  ✓ Encoded '{col}': {len(le.classes_)} unique values")
    
    # ==========================================
    # STEP 8: SPLIT DATA INTO TRAIN AND TEST SETS
    # ==========================================
    
    # Split: 80% for training, 20% for testing
    # random_state=42 ensures reproducibility (same split every time)
    # stratify=y ensures both sets have similar class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,        # 20% for testing
        random_state=42,      # For reproducibility
        stratify=y            # Maintain class balance
    )
    
    print(f"\n✂️  DATA SPLIT:")
    print(f"  Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X))*100:.1f}%)")
    print(f"  Testing set:  {X_test.shape[0]} samples ({(X_test.shape[0]/len(X))*100:.1f}%)")
    
    # Show class distribution in splits
    print(f"\n📊 Target distribution in training set:")
    unique, counts = pd.Series(y_train).value_counts().sort_index().items()
    for val, count in zip(*pd.Series(y_train).value_counts().sort_index().items()):
        print(f"  Class {val}: {count} samples")
    
    # ==========================================
    # STEP 9: FEATURE SCALING
    # ==========================================
    
    scaler = StandardScaler()
    
    # Fit scaler on training data only (to avoid data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply same transformation to test data
    # (uses mean and std from training data)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n⚖️  FEATURE SCALING COMPLETED:")
    print(f"  Features standardized to mean=0, std=1")
    print(f"  Training set shape: {X_train_scaled.shape}")
    print(f"  Test set shape: {X_test_scaled.shape}")
    
    # ==========================================
    # STEP 10: RETURN PROCESSED DATA
    # ==========================================
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Return all the processed components
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    """
    This block runs when the script is executed directly.
    It calls the preprocessing function and displays the results.
    """
    
    print("\n" + "🚀 " * 20)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("🚀 " * 20 + "\n")
    
    # Run the preprocessing pipeline
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    # Display final summary
    print("\n" + "📋 " * 20)
    print("FINAL SUMMARY:")
    print(f"  • Feature names: {len(feature_names)} features")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")
    print(f"  • Scaler fitted and ready for future use")
    print("📋 " * 20 + "\n")
    
    print("✨ Data is now ready for model training! ✨\n")