import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(data):
    print(data.isnull().sum())
    data = data.ffill()
    return data

def detect_anomalies(data, column):
    model = IsolationForest(n_estimators=100, contamination=float(.01))
    data['anomaly'] = model.fit_predict(data[[column]])
    anomalies = data[data['anomaly'] == -1]
    return anomalies

def analyze_attack_types(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Attack Type')
    plt.title('Розподіл типів атак')
    plt.show()

def correlation_analysis(data):
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Кореляційна матриця')
    plt.show()

def visualize_data(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Packet Length', bins=30, kde=True)
    plt.title('Розподіл довжини пакетів')
    plt.show()

if __name__ == '__main__':
    data = load_data('cybersecurity_attacks.csv')
    data = clean_data(data)
    anomalies = detect_anomalies(data, 'Packet Length')
    analyze_attack_types(data)
    correlation_analysis(data)
    visualize_data(data)