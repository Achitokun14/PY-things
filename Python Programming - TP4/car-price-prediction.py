import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Creation
def create_dataset(n_samples=1000):
    car_brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']
    
    data = {
        'brand': np.random.choice(car_brands, n_samples),
        'age': np.random.randint(0, 20, n_samples),
        'mileage': np.random.normal(50000, 20000, n_samples),
        'engine_size': np.random.choice([1.4, 1.6, 1.8, 2.0, 2.4, 3.0], n_samples)
    }
    
    # Create synthetic prices based on features
    base_price = 20000
    brand_premium = {'Toyota': 0, 'Honda': 1000, 'Ford': -1000, 'BMW': 15000, 'Mercedes': 20000}
    
    prices = []
    for i in range(n_samples):
        price = (base_price 
                + brand_premium[data['brand'][i]]
                - data['age'][i] * 1000
                - data['mileage'][i] * 0.1
                + data['engine_size'][i] * 5000
                + np.random.normal(0, 2000))
        prices.append(max(price, 5000))  # Ensure no negative prices
    
    data['price'] = prices
    return pd.DataFrame(data)

# Step 2: Data Exploration
def explore_data(df):
    print("Basic Data Information:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='price')
    plt.title('Distribution of Car Prices')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='brand', y='price')
    plt.title('Price Distribution by Brand')
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='age', y='price')
    plt.title('Price vs Age')
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='mileage', y='price')
    plt.title('Price vs Mileage')
    
    plt.tight_layout()
    plt.show()

# Step 3: Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df['mileage'] = df['mileage'].fillna(df['mileage'].mean())
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['brand'], prefix='brand')
    return df_encoded

# Step 4: Model Development
def build_model(df_encoded):
    # Split features and target
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, X.columns

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test, feature_names):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    feature_importance = feature_importance.sort_values(
        'Coefficient', key=abs, ascending=False
    )
    print("\nFeature Importance:")
    print(feature_importance)
    
    return r2, mse, rmse, feature_importance

# Step 6: Make Predictions
def predict_price(model, sample_data):
    prediction = model.predict(sample_data)
    return prediction

# Main execution
if __name__ == "__main__":
    # Create dataset
    df = create_dataset()
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    df_encoded = preprocess_data(df)
    
    # Build model
    model, X_train, X_test, y_train, y_test, feature_names = build_model(df_encoded)
    
    # Evaluate model
    r2, mse, rmse, feature_importance = evaluate_model(model, X_test, y_test, feature_names)
    
    # Example prediction
    new_sample = pd.DataFrame({
        'age': [5],
        'mileage': [45000],
        'engine_size': [2.0],
        'brand_BMW': [0],
        'brand_Ford': [0],
        'brand_Honda': [0],
        'brand_Mercedes': [0],
        'brand_Toyota': [1]
    })
    
    predicted_price = predict_price(model, new_sample)
    print(f"\nPredicted price for sample car: ${predicted_price[0]:,.2f}")
