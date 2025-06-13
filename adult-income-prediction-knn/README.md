
# Adult Dataset Analysis and KNN Model Application

## Overview

This project involves data preprocessing and applying a K-Nearest Neighbors (KNN) classification model on the "Adult" dataset. The dataset is used to predict whether an individual earns more than $50K/year based on various demographic factors.

## Project Structure

- `adult.csv`: Original dataset in CSV format.
- `pre_process.ipynb`: Jupyter notebook containing data cleaning, preprocessing, and feature engineering steps.
- `cleaned_adult.csv`: Cleaned dataset after preprocessing.
- `knn_model.ipynb`: Jupyter notebook containing the application of the KNN model and evaluation metrics.

## Steps

### 1. Data Preprocessing (`pre_process.ipynb`)

In this notebook, the following preprocessing steps are applied to the "Adult" dataset:

- **Loading the Dataset**: The original dataset is loaded and examined.
- **Missing Values Handling**: Missing values are handled either by removing or filling them.
- **Encoding Categorical Features**: Categorical variables are encoded using techniques like one-hot encoding or label encoding.
- **Feature Scaling**: Numerical features are scaled using StandardScaler to ensure they are on the same scale for KNN.
- **Dataset Splitting**: The dataset is split into training and test sets for model evaluation.

After running the preprocessing steps, the cleaned dataset is saved as `adult_cleaned.csv`.

### 2. KNN Model Application (`knn_model.ipynb`)

This notebook covers the application of the K-Nearest Neighbors (KNN) classification algorithm:

- **Model Training**: The KNN classifier is trained using the cleaned dataset (`adult_cleaned.csv`).
- **Model Tuning**: Hyperparameter tuning is done to select the optimal number of neighbors (`k`).
- **Model Evaluation**: The model is evaluated using accuracy, precision, recall, and F1-score metrics.

### Results

- **Accuracy**: 0.8352
- **Precision, Recall, F1-Score**: Detailed classification metrics are provided in the notebook.

### Requirements

To run the notebooks, install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

Here’s the list of required libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Example Usage

1. **Preprocessing**: Open and run the `adult_preprocessing.ipynb` notebook to clean and preprocess the dataset.
2. **KNN Model**: Open and run the `knn_model.ipynb` notebook to train the KNN classifier and evaluate its performance.

### Conclusion

This project demonstrates how to preprocess the "Adult" dataset, apply machine learning algorithms like KNN, and evaluate model performance. The final result shows the model's accuracy and other performance metrics.

