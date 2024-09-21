# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the cleaned customer data
data = pd.read_csv('../datasets/cleaned_customer_data.csv')

# Separate features (X) and target variable (y)
X = data.drop(['total_spent'], axis=1)
y = data['total_spent']  # No need for log transformation with Random Forest

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on train and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate MSE for train and test sets
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Print sample predictions
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {y_test_pred[i]:.2f}")

# Calculate and print Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
