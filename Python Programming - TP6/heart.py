import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Read data
def read_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        print(f"Error: File {url} not found.")
        return None

# 2. Explore data
def explore(df):
    if df is None:
        return

    print(f"Shape of data: {df.shape}")
    print(50 * "-")
    print("\nDescribe data:\n")
    print(df.describe())
    print(50 * "-")
    print("\nGeneral info:\n")
    df.info()
    print(50 * "-")
    print(f"Missing values:\n{df.isna().sum()}")

    print(f"HeartDisease distribution:\n{df['HeartDisease'].value_counts(normalize=True)}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    df['HeartDisease'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of HeartDisease")
    
    plt.subplot(1, 2, 2)
    cols_num = df.select_dtypes(include="number").columns
    sns.heatmap(df[cols_num].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def handle_outlier(df):
    cols_num = df.select_dtypes(include="number").columns
    df2 = df.copy()
    for col in cols_num:
        z_score = (df2[col] - df2[col].mean()) / df2[col].std()
        df2 = df2[abs(z_score) < 3]
    return df2

def clean(df):
    df2 = df.copy()
    df2.drop_duplicates(inplace=True)
    
    # Optional columns to drop (adjust based on your dataset)
    cols_to_drop = ["Slope", "ST_Slope"]
    df2.drop(columns=[col for col in cols_to_drop if col in df2.columns], inplace=True)

    # Drop highly correlated columns
    cols_num = df2.select_dtypes(include="number").columns
    correlated_pairs = []
    for col1 in cols_num:
        for col2 in cols_num:
            if col1 != col2 and abs(df2[col1].corr(df2[col2])) > 0.9:
                correlated_pairs.append((col1, col2))
    
    unique_cols = set([x for pair in correlated_pairs for x in pair])
    df2.drop(columns=list(unique_cols), inplace=True)

    return df2

def transform(df):
    df2 = df.copy()
    categorical_cols = df2.select_dtypes(include="object").columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        df2[col] = encoder.fit_transform(df2[col])
    return df2

def scale(df):
    df2 = df.copy()
    scaler = StandardScaler()
    cols_num = df2.select_dtypes(include="number").columns
    df2[cols_num] = scaler.fit_transform(df2[cols_num])
    df2['HeartDisease'] = df2['HeartDisease'].astype(int)
    return df2

def split(df):
    df2 = df.copy()
    x = df2.drop('HeartDisease', axis=1)
    y = df2['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE()
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    return x_train_smote, x_test, y_train_smote, y_test

def train_models(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression()
    rf = RandomForestClassifier()

    print("KNN:")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("Logistic Regression:")
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("Random Forest:")
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

def main():
    url = "heart.csv"
    df = read_data(url)

    if df is not None:
        print(df.head())
        df2 = handle_outlier(df)
        print(df2.head())
        print(f"Original shape: {df.shape}, After outlier removal: {df2.shape}")
        
        df3 = clean(df2)
        print(f"After cleaning: {df3.shape}")
        df4 = transform(df3)
        print(f"After transformation: {df4.head()}")

        df5 = scale(df4)
        print(f"After scaling: {df5.head()}")

        x_train_smote, x_test, y_train_smote, y_test = split(df5)
        train_models(x_train_smote, y_train_smote, x_test, y_test)

if __name__ == "__main__":
    main()