# Data_Analysis.py – Iris Dataset Exploratory Analysis

## Overview

`Data_Analysis.py` is a Python script for performing exploratory data analysis (EDA) and visualization on the classic Iris dataset. The script demonstrates core data science workflows, including data loading, cleaning, statistical analysis, and generating visualizations to uncover insights about the dataset.

## Features

- **Loads the Iris dataset** from scikit-learn.
- **Explores structure and missing data**.
- **Cleans the dataset** (removes missing values, if any).
- **Computes basic statistics** for each feature.
- **Maps numeric target to species names** for interpretability.
- **Groups by species** to display mean statistics.
- **Highlights key findings** about the dataset.
- **Visualizes data** with Matplotlib and Seaborn:
  - Sepal Length trends by species (line chart)
  - Average Petal Length by species (bar chart)
  - Sepal Width distribution (histogram)
  - Sepal Length vs Petal Length scatter plot

## Usage

### Requirements

- Python 3.x
- Required libraries (install with pip if needed):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Run the script

```bash
python Data_Analysis.py
```

The script will display outputs in the console and show plots in a window.

## Script Structure

1. **Data Loading & Exploration**  
   Loads the Iris dataset, creates a DataFrame, checks for missing values, and displays basic info.

2. **Data Cleaning**  
   Drops any missing values (Iris typically has none).

3. **Basic Data Analysis**  
   Provides summary statistics, maps target values to species names, groups by species, and prints findings.

4. **Data Visualization**  
   Produces four different charts to illustrate relationships and distributions in the data.

## Output

- Console printouts of data structure, statistics, and insights.
- Four visualizations for further exploration.

## Error Handling

- If the dataset cannot be loaded, a relevant error message is displayed.
- Other unexpected errors are caught and printed.

## License

This script is provided for educational and demonstration purposes. See repository license for details.

## Author

[Agabe-Dev](https://github.com/Agabe-Dev)
