# Decision Tree with Custom Pruning

This project implements a decision tree from scratch with custom pruning techniques. It includes two main scripts: one for exploratory data analysis (EDA) and another for the decision tree implementation and evaluation.

## Project Structure

- `EDA.py`: This script performs exploratory data analysis on the Wine dataset, visualizing relationships between features and the target variable.
- `main.py`: This script contains the implementation of a decision tree classifier with custom pruning methods. It includes functions for training, predicting, and pruning the decision tree.

## Dataset

The project uses the Wine dataset from the `sklearn.datasets` module. This dataset is a classic and very easy multi-class classification dataset.

## Features

- **Decision Tree Implementation**: A custom decision tree classifier built from scratch.
- **Pruning Techniques**: Includes both reduced error pruning and a custom pruning method based on statistical thresholds.
- **Exploratory Data Analysis**: Visualizes the dataset to understand feature distributions and relationships.

## Requirements

- Python 3.12.4
- Required Python packages are listed in `requirements.txt`. You can install them using:

  ```bash
  pip install -r requirements.txt
