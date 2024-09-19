import pandas as pd
import numpy as np
from datetime import datetime

def print_df_info(df, step):
    print(f"\n--- After {step} ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")

# Read the CSV file
df = pd.read_csv("../datasets/MOCK_DATA.csv")
print_df_info(df, "reading CSV")

# Convert date columns to datetime
date_columns = ['signup_date', 'last_purchase_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
print_df_info(df, "converting dates")

# Remove rows where last_purchase_date is before signup_date
df = df[df['last_purchase_date'] >= df['signup_date']]
print_df_info(df, "removing invalid date ranges")

# Remove rows with future dates
current_date = datetime.now()
df = df[(df['signup_date'] <= current_date) & (df['last_purchase_date'] <= current_date)]
print_df_info(df, "removing future dates")

# Calculate and update total_spent
df['total_spent'] = df['total_orders'] * df['average_order_value']
print_df_info(df, "updating total_spent")

# Update frequency_of_purchases
df['frequency_of_purchases'] = (df['last_purchase_date'] - df['signup_date']).dt.days / df['total_orders']
print_df_info(df, "updating frequency_of_purchases")

# Remove rows with unreasonable ages
df = df[(df['age'] >= 18) & (df['age'] <= 100)]
print_df_info(df, "removing unreasonable ages")

# Remove rows with unreasonably high total_orders or average_order_value
df = df[df['total_orders'] <= 1000]
df = df[df['average_order_value'] <= 10000]
print_df_info(df, "removing unreasonable orders and values")

# Remove rows with unreasonably high website_visits
df = df[df['website_visits'] <= 10000]
print_df_info(df, "removing unreasonable website visits")

# Handle email_open_rate
if 'email_open_rate ' in df.columns:
    df['email_open_rate'] = df['email_open_rate '].astype(float)
    df = df[df['email_open_rate'] <= 100]
    df['email_open_rate'] = df['email_open_rate'].fillna(df['email_open_rate'].mean())
    df = df.drop('email_open_rate ', axis=1)
print_df_info(df, "handling email_open_rate")

# Fill missing values
df['social_media_engagement'] = df['social_media_engagement'].fillna(df['social_media_engagement'].median())
print_df_info(df, "filling missing values")

# Drop rows with any remaining missing values
df = df.dropna()
print_df_info(df, "dropping rows with missing values")

# Create new features from date columns
df['account_age_days'] = (current_date - df['signup_date']).dt.days
df['days_since_last_purchase'] = (current_date - df['last_purchase_date']).dt.days
df = df.drop(columns=['signup_date', 'last_purchase_date'])

# Encode categorical variables
categorical_cols = ['gender', 'product_categories', 'payment_method', 'device_type', 'loyalty_program_status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print_df_info(df, "encoding categorical variables and creating new features")

# Remove the 'id' column
df = df.drop('id', axis=1)

# Reset the index
df = df.reset_index(drop=True)

# Display the first few rows of the cleaned dataset
print(df.head())

# Save the cleaned dataset
df.to_csv('../datasets/cleaned_customer_data.csv', index=False)
print("\nCleaned data saved to '../datasets/cleaned_customer_data.csv'")
