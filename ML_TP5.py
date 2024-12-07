# import 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# 1. read data
def read_data(url):
    df = pd.read_csv(url)
    return df

# 2.explore data

def explore(df):
    print(f"shape of data {df.shape} ")
    print(50*"-")
    print("\nDescribe data\n")
    print(df.describe())
    print(50*"-")
    print("\n General info\n")
    print(df.info())
    print(50*"-")
    print(f"missing values : {df.isna().sum()} ")

    print(f"Churn distribution : {df['Churn'].value_counts(normalize=True)}")
    plt.figure()
    plt.pie(df['Churn'].value_counts(),labels=df['Churn'].value_counts().index)
    plt.title("Distribution of Churn")
    plt.show()

    #heatmap

    cols_num=df.select_dtypes(include="number").columns
    sns.heatmap(df[cols_num].corr(), annot=True, cmap="coolwarm")
    plt.show()

def handle_outlier(df):
    cols_num=df.select_dtypes(include="number").columns
    for col in cols_num:
        z_score=(df[col]-df[col].mean())/df[col].std()
        df2 = df[abs(z_score)<3]
    return df2

def clean(df):
    df.drop_duplicates(inplace=True)
    cols_to_drop=["State","Area code"]
    df.drop(columns= cols_to_drop, inplace=True)
    cols_num=df.select_dtypes(include="number").columns
    correlated=[]
    
    for col1 in cols_num :
        l=[]
        for col2 in cols_num:
            if  df[col1].corr(df[col2]) > 0.9 :
                l.append(col2)
        if len(l)>1 and l not in correlated:
            correlated.append(l)
    first_element = [col[0] for col in correlated]
    df = df.drop(columns=first_element)
    return df
    

def transform(df):
    categorical_col = df.select_dtypes(include="object").columns
    
    for col in categorical_col:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    return df

def scale(df):
    scaler= StandardScaler()
    cols_num=df.select_dtypes(include="number").columns
    df[cols_num]=scaler.fit_transform(df[cols_num])
    df['Churn']= df['Churn'].astype(int)
    return df

def split(df):
    x=df.drop('Churn',axis=1)
    y=df['Churn']
    x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)
    smote= SMOTE()
    x_train_smote, y_train_smote = smote.fit_resample(x_train,y_train)
    return x_train_smote,x_test,y_train_smote, y_test

def train_models(x_train, y_train,x_test,y_test):
    
    KNN =KNeighborsClassifier(n_neighbors=3)
    LR=LogisticRegression()
    RF =RandomForestClassifier()
    print("KNN")
    KNN.fit(x_train, y_train)
    y_pred= KNN.predict(x_test)
    print(classification_report(y_pred=y_pred,y_true=y_test))
    print("lg")
    LR.fit(x_train, y_train)
    y_pred= LR.predict(x_test)
    print(classification_report(y_pred=y_pred,y_true=y_test))
    print("RF")
    RF.fit(x_train, y_train)
    y_pred= RF.predict(x_test)
    print(classification_report(y_pred=y_pred,y_true=y_test))


def main():

    url = "churn-bigml-20.csv"
    df = read_data(url)

    print(df.head())

    #explore(df)
    df2=handle_outlier(df)
    print(df2.head())
    print(df.shape)
    print(df2.shape)
    
    df3= clean(df2)
    print(df3.shape)
    df3=transform(df3)
    print(df3.head())

    df3 = scale(df3)
    print(df3.head())

    x_train_smote,x_test,y_train_smote, y_test = split(df3)

    train_models(x_train_smote,y_train_smote,x_test, y_test)


if __name__ == "__main__":
    main()