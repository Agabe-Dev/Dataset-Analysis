import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    try:
        # Task 1: Load and Explore the Dataset
        print("=== TASK 1: LOAD AND EXPLORE DATASET ===")
        iris = load_iris()
        iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                              columns=iris['feature_names'] + ['target'])
        
        print("\nFirst 5 rows of the dataset:")
        print(iris_df.head())
        
        print("\nDataset structure:")
        print(iris_df.info())
        
        print("\nMissing values:")
        print(iris_df.isnull().sum())
        
        # Clean dataset (though Iris typically has no missing values)
        iris_df.dropna(inplace=True)
        print("\nDataset after cleaning:")
        print(iris_df.isnull().sum())
        
        # Task 2: Basic Data Analysis
        print("\n\n=== TASK 2: BASIC DATA ANALYSIS ===")
        print("\nBasic statistics of numerical columns:")
        print(iris_df.describe())
        
        # Add species names for better interpretation
        iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("\nMean values by species:")
        grouped_stats = iris_df.groupby('species').mean()
        print(grouped_stats)
        
        print("\nInteresting findings:")
        print("- Setosa has significantly shorter petal length and width compared to others")
        print("- Versicolor and virginica are more similar but can be distinguished by petal measurements")
        print("- Sepal width shows less variation across species compared to other features")
        
        # Task 3: Data Visualization
        print("\n\n=== TASK 3: DATA VISUALIZATION ===")
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 12))
        
        # 1. Line chart (simulating trends over time by using index as time)
        plt.subplot(2, 2, 1)
        for species in iris_df['species'].unique():
            species_data = iris_df[iris_df['species'] == species]
            plt.plot(species_data.index, species_data['sepal length (cm)'], 
                     label=f'Sepal Length - {species}')
        plt.title('Sepal Length Trends by Species')
        plt.xlabel('Observation Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        
        # 2. Bar chart - average petal length per species
        plt.subplot(2, 2, 2)
        sns.barplot(x='species', y='petal length (cm)', data=iris_df, estimator=np.mean)
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Mean Petal Length (cm)')
        
        # 3. Histogram - sepal width distribution
        plt.subplot(2, 2, 3)
        sns.histplot(iris_df['sepal width (cm)'], bins=15, kde=True)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        
        # 4. Scatter plot - sepal length vs petal length
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', 
                        hue='species', data=iris_df)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Error: Dataset file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
