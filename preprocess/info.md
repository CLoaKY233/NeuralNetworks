# Data Cleaning and Preprocessing Script Documentation

## Overview

This script performs comprehensive data cleaning and preprocessing on a customer dataset. It handles various data quality issues and prepares the dataset for further analysis or machine learning tasks.

## Key Functions and Steps

### 1. Helper Function: `print_df_info(df, step)`

```python
def print_df_info(df, step):
    print(f"\n--- After {step} ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")
```

This function prints information about the DataFrame after each processing step.

### 2. Data Loading and Initial Processing

| Step | Action | Description |
|------|--------|-------------|
| 1 | Read CSV | Loads the "MOCK_DATA.csv" file into a pandas DataFrame |
| 2 | Convert Dates | Transforms 'signup_date' and 'last_purchase_date' to datetime format |

### 3. Data Cleaning

#### Date-related Cleaning

- Remove rows where:
  - 'last_purchase_date' is before 'signup_date'
  - 'signup_date' or 'last_purchase_date' is in the future

#### Numerical Data Cleaning

| Field | Action |
|-------|--------|
| `total_spent` | Recalculate: `total_orders * average_order_value` |
| `frequency_of_purchases` | Recalculate: Average days between purchases |
| `age` | Remove if not between 18 and 100 |
| `total_orders` | Remove if > 1000 |
| `average_order_value` | Remove if > 10000 |
| `website_visits` | Remove if > 10000 |

#### Handling Specific Fields

- `email_open_rate`:
  - Convert to float
  - Remove if > 100%
  - Fill missing values with mean

- `social_media_engagement`:
  - Fill missing values with median

### 4. Feature Engineering

- Create new features:
  - `account_age_days`
  - `days_since_last_purchase`
- Drop original date columns

### 5. Categorical Data Encoding

- Apply one-hot encoding to:
  - `gender`
  - `product_categories`
  - `payment_method`
  - `device_type`
  - `loyalty_program_status`

### 6. Final Data Preparation

- Remove 'id' column
- Reset DataFrame index

## Output

- Display first few rows of cleaned dataset
- Save cleaned data to '../datasets/cleaned_customer_data.csv'

---

## Code Snippet: Categorical Encoding

```python
categorical_cols = ['gender', 'product_categories', 'payment_method', 'device_type', 'loyalty_program_status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

---

> **Note**: This script significantly transforms the original dataset. Ensure to validate the results and adjust parameters as needed for your specific use case.
