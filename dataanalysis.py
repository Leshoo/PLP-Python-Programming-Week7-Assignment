import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Data
def load_data():
    # Sample dataset (replace with actual dataset if available)
    data = {
        "Time": [1, 2, 3, 4, 5],
        "SepalLength": [5.1, 4.9, 4.7, 5.0, 5.2],
        "SepalWidth": [3.5, 3.0, 3.2, 3.1, 3.6],
        "PetalLength": [1.4, 1.4, 1.3, 1.5, 1.6],
        "Species": ["setosa", "setosa", "setosa", "setosa", "setosa"]
    }
    return pd.DataFrame(data)

# Data Exploration
def explore_data(df):
    print("First few rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

# Visualizations
def create_visualizations(df):
    # Line Chart
    plt.figure(figsize=(10, 6))
    plt.plot(df["Time"], df["SepalLength"], marker='o', color='blue', label='Sepal Length')
    plt.title('Sepal Length Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Sepal Length', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Bar Chart
    avg_petal_length = df.groupby('Species')['PetalLength'].mean()
    plt.figure(figsize=(8, 6))
    avg_petal_length.plot(kind='bar', color='green')
    plt.title('Average Petal Length per Species', fontsize=16)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Average Petal Length', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["SepalWidth"], bins=5, color='orange', edgecolor='black')
    plt.title('Distribution of Sepal Width', fontsize=16)
    plt.xlabel('Sepal Width', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y')
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df["SepalLength"], df["PetalLength"], color='purple', label='Data Points')
    plt.title('Sepal Length vs. Petal Length', fontsize=16)
    plt.xlabel('Sepal Length', fontsize=12)
    plt.ylabel('Petal Length', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
def main():
    print("Loading Data...")
    df = load_data()
    
    print("\nExploring Data...")
    explore_data(df)
    
    print("\nCreating Visualizations...")
    create_visualizations(df)
    
    print("\nFindings:")
    print("- Sepal Length remains consistent over time in this sample.")
    print("- The average petal length for the species is similar due to uniform data.")
    print("- Sepal Width is normally distributed in the dataset.")
    print("- There is a positive relationship between Sepal Length and Petal Length.")

if __name__ == "__main__":
    main()
