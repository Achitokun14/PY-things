import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Exploration and Preprocessing

# Load the data
df = pd.read_csv('5G_energy_consumption_dataset.csv')  # Using the data from the image

# 1.1 Basic Data Exploration
def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    return df

# 1.2 Data Cleaning
def clean_data(df):
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Convert time column to datetime
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d %H%M%S')
    
    # Handle potential outliers using IQR method for 'Energy' column
    Q1 = df['Energy'].quantile(0.25)
    Q3 = df['Energy'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Energy'] < (Q1 - 1.5 * IQR)) | (df['Energy'] > (Q3 + 1.5 * IQR)))]
    
    return df

# 1.3 Feature Engineering
def engineer_features(df):
    # Extract time-based features
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    
    # Create binary features for high/low load periods
    df['high_load'] = (df['load'] > df['load'].median()).astype(int)
    
    # One-hot encode BS column if needed
    df = pd.get_dummies(df, columns=['BS'], prefix='bs')
    
    return df

# 2. Model Development
def build_model(df):
    # 2.1 Feature Selection
    features = ['load', 'hour', 'day', 'high_load'] + [col for col in df.columns if col.startswith('bs_')]
    X = df[features]
    y = df['Energy']
    
    # 2.2 Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2.3 Model Implementation
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# 3. Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Create visualization of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Energy Consumption')
    plt.ylabel('Predicted Energy Consumption')
    plt.title('Actual vs Predicted Energy Consumption')
    plt.tight_layout()
    
    return r2, mse

# Main execution
if __name__ == "__main__":
    # Execute the pipeline
    df = explore_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    model, X_train, X_test, y_train, y_test = build_model(df)
    r2, mse = evaluate_model(model, X_test, y_test)
    
    # Test with custom input
    sample_input = X_test.iloc[0].to_dict()
    print("\nSample Prediction:")
    print(f"Input features: {sample_input}")
    prediction = model.predict(X_test.iloc[[0]])
    print(f"Predicted energy consumption: {prediction[0]:.2f}")
