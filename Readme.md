# Customer Spending Prediction Models

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green)
![Pandas](https://img.shields.io/badge/Pandas-Latest-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Latest-lightgrey)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
  - [Deep Neural Network](#deep-neural-network)
  - [Ensemble Model](#ensemble-model)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Project Overview

This repository contains machine learning and deep learning models designed to predict customer spending patterns. The project demonstrates the application of data preprocessing, feature engineering, and model development to forecast customer expenditure.

## Dataset

The project uses a custom dataset (`MOCK_DATA.csv`) containing customer information. This data is cleaned and preprocessed to create `cleaned_customer_data.csv`, which is used for training and testing our models.

## Models

### Deep Neural Network

Our deep learning model uses TensorFlow and Keras:

- Multiple dense layers with dropout for regularization
- ReLU activation for hidden layers, linear activation for output layer
- Adam optimizer
- L2 regularization and dropout to prevent overfitting

For details, see [Deep Learning Model Details](keras/info.md).

### Ensemble Model

The ensemble model combines multiple algorithms:

- Random Forest, Gradient Boosting, and Neural Network regressors
- Voting Regressor as the ensemble method

For more information, refer to [Ensemble Model Details](scikitlearn/info.md).

## Project Structure

```
.
├── datasets/
│   ├── MOCK_DATA.csv
│   └── cleaned_customer_data.csv
├── keras/
│   ├── deeplearn.py
│   └── info.md
├── scikitlearn/
│   ├── ensemble.py
│   └── info.md
├── preprocess/
│   ├── datacleaner.py
│   └── info.md
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/customer-spending-prediction.git
   cd customer-spending-prediction
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Data Preprocessing:
   ```
   python preprocess/datacleaner.py
   ```

2. Run Deep Learning Model:
   ```
   python keras/deeplearn.py
   ```

3. Run Ensemble Model:
   ```
   python scikitlearn/ensemble.py
   ```

## Results

The performance metrics for each model are printed after running the respective scripts. These include Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).

## Documentation

- [Data Cleaning and Preprocessing](preprocess/info.md)
- [Deep Learning Model Information](keras/info.md)
- [Ensemble Model Information](scikitlearn/info.md)

## Contributing

Contributions to improve the project are welcome. Please fork the repository and submit a pull request with your proposed changes.

---
