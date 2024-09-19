import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv('../datasets/cleaned_customer_data.csv')

# Prepare the features (X) and target (y)
X = data.drop(['total_spent'], axis=1)
y = np.log1p(data['total_spent'])  # Log transform the target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
inputs = Input(shape=(X_train.shape[1],))
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(inputs)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Evaluate the model
train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Train MSE: {train_loss:.2f}")
print(f"Test MSE: {test_loss:.2f}")

# Make predictions
y_pred = np.expm1(model.predict(X_test_scaled))
y_test_original = np.expm1(y_test)

# Print some sample predictions
for i in range(5):
    print(f"Actual: {y_test_original.iloc[i]:.2f}, Predicted: {y_pred[i][0]:.2f}")

# Calculate and print MAPE
mape = np.mean(np.abs((y_test_original - y_pred.flatten()) / y_test_original)) * 100
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
