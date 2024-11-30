import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Step 1: Data Loading and Exploration
# Load the churn dataset into a pandas DataFrame
df = pd.read_csv('churn-bigml-20.csv')

# Display the first few rows to understand the data structure
print(df.head())

# Print the shape of the dataset
print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')

# Identify and handle missing values
print(df.isnull().sum())  # Check for any null values
print(df.describe())  # Summarize the key statistics for each column

# Analyze the distribution of the target variable (Churn)
print(df['Churn'].value_counts())
df['Churn'].value_counts().plot(kind='bar')
plt.show()

# Visualize relationships between features using heatmaps and pair plots
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
plt.show()

sns.pairplot(df)
plt.show()

# Step 2: Data Cleaning and Preprocessing
# Check percentage of missing values
print(df.isnull().mean() * 100)

# Impute missing values using median
df = df.fillna(df.median())

# Detect and handle outliers using box plots
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(16,12))
plt.show()

# Cap outliers at the 99th percentile
for col in df.select_dtypes(include='number'):
    df[col] = np.where(df[col] > df[col].quantile(0.99), df[col].quantile(0.99), df[col])

# Check for and remove duplicate rows
print(f'Number of duplicate rows: {df.duplicated().sum()}')
df = df.drop_duplicates()

# Drop unnecessary columns
df = df.drop(['Column1', 'Column2'], axis=1)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['International plan', 'Voice mail plan'])

# Scale numeric features
numeric_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 3: Model Training and Evaluation
# Split the dataset into training and testing sets
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate the models on the test set
y_pred_knn = knn.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

print(f'KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}')
print(f'KNN Precision: {precision_score(y_test, y_pred_knn)}')
print(f'KNN Recall: {recall_score(y_test, y_pred_knn)}')
print(f'KNN F1-score: {f1_score(y_test, y_pred_knn)}')

print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print(f'Logistic Regression Precision: {precision_score(y_test, y_pred_lr)}')
print(f'Logistic Regression Recall: {recall_score(y_test, y_pred_lr)}')
print(f'Logistic Regression F1-score: {f1_score(y_test, y_pred_lr)}')

print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Random Forest Precision: {precision_score(y_test, y_pred_rf)}')
print(f'Random Forest Recall: {recall_score(y_test, y_pred_rf)}')
print(f'Random Forest F1-score: {f1_score(y_test, y_pred_rf)}')

# Step 4: Model Comparison and Selection
# Create a table to compare model performance
model_performance = {
    'Model': ['KNN', 'Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
    'Precision': [precision_score(y_test, y_pred_knn), precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf)],
    'Recall': [recall_score(y_test, y_pred_knn), recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
    'F1-score': [f1_score(y_test, y_pred_knn), f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)]
}
performance_df = pd.DataFrame(model_performance)
print(performance_df)


'''
results in terminal :


'''