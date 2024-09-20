# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the cleaned customer data
data = pd.read_csv('../datasets/cleaned_customer_data.csv')

# Separate features (X) and target variable (y)
X = data.drop(['total_spent'], axis=1)
y = np.log1p(data['total_spent'])  # Log transform to handle skewed distribution

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define individual models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# Create the ensemble model
ensemble_model = VotingRegressor([
    ('rf', rf_model),
    ('gb', gb_model),
    ('nn', nn_model)
])

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = np.expm1(ensemble_model.predict(X_test_scaled))  # Reverse log transformation
y_test_original = np.expm1(y_test)

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test_original, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate and print Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Print sample predictions
for i in range(5):
    print(f"Actual: {y_test_original.iloc[i]:.2f}, Predicted: {y_pred[i]:.2f}")
