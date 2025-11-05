import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/content/Cashflow_dataset.csv')
    # Handle missing values
    numeric_cols = [
        'Revenue', 'COGS', 'Operating_Expenses','Depreciation_Amortization',
        'Change_in_Inventory','Accounts_Receivable', 'Accounts_Payable', 'Taxes_Paid',
        'CapEx', 'Asset_Sale_Proceeds', 'Investments_Bought',
        'Investments_Sold','Interest_Received',
        'Debt_Raised', 'Debt_Repaid', 'Interest_Paid',
        'Equity_Issued', 'Dividends_Paid', 'Net_Cash_Flow',
    ]
    categorical_cols = ['Company_Name',  'Month']

    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Feature Engineering - Cash Flows
    data['Operating_Cash_Flow'] = (
        (data['Revenue'] - data['COGS'] - data['Operating_Expenses'])
        + data['Depreciation_Amortization']
        - (data['Change_in_Inventory'] + data['Accounts_Receivable'] - data['Accounts_Payable'])
        - data['Taxes_Paid']
    )

    data['Investing_Cash_Flow'] = (
        (-data['CapEx'])
        + data['Asset_Sale_Proceeds']
        - data['Investments_Bought']
        + data['Investments_Sold']
        + data['Interest_Received']
    )

    data['Financing_Cash_Flow'] = (
        data['Debt_Raised'] - data['Debt_Repaid'] - data['Interest_Paid']
        + data['Equity_Issued'] - data['Dividends_Paid']
    )

    data['Cash_Flow'] = data['Operating_Cash_Flow'] + data['Investing_Cash_Flow'] + data['Financing_Cash_Flow']

    return data, numeric_cols, categorical_cols

data, numeric_cols, categorical_cols = load_data()

# Encoding and Scaling
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                            columns=encoder.get_feature_names_out(categorical_cols))
data_encoded = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                          encoded_cols.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

X = data_encoded.drop(columns=['Net_Cash_Flow'])
y = data_encoded['Net_Cash_Flow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model (best performing)
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

def calculate_cash_flows(input_dict):
    # Calculate operating, investing, financing and net cash flow from user input
    operating_cash_flow = (
        (input_dict['Revenue'] - input_dict['COGS'] - input_dict['Operating_Expenses'])
        + input_dict['Depreciation_Amortization']
        - (input_dict['Change_in_Inventory'] + input_dict['Accounts_Receivable'] - input_dict['Accounts_Payable'])
        - input_dict['Taxes_Paid']
    )
    investing_cash_flow = (
        (-input_dict['CapEx'])
        + input_dict['Asset_Sale_Proceeds']
        - input_dict['Investments_Bought']
        + input_dict['Investments_Sold']
        + input_dict['Interest_Received']
    )
    financing_cash_flow = (
        input_dict['Debt_Raised']
        - input_dict['Debt_Repaid']
        - input_dict['Interest_Paid']
        + input_dict['Equity_Issued']
        - input_dict['Dividends_Paid']
    )
    net_cash_flow = operating_cash_flow + investing_cash_flow + financing_cash_flow
    return operating_cash_flow, investing_cash_flow, financing_cash_flow, net_cash_flow

# Streamlit User Interface
st.title("Cash Flow Prediction & Calculator")

st.header("Enter Financial Data")
with st.form("input_form"):
    user_input = {}
    for col in numeric_cols:
        if col != 'Net_Cash_Flow':
            user_input[col] = st.number_input(col, value=0.0, format="%.2f")

    submitted = st.form_submit_button("Calculate Cash Flows and Predict Net Cash Flow")

if submitted:
    # Calculate cash flows
    op_cf, inv_cf, fin_cf, net_cf_calc = calculate_cash_flows(user_input)

    # Prepare user input for prediction model
    user_df = pd.DataFrame([user_input])
    # Encode categorical columns from data to get columns names and align dummy columns for input
    # As categorical input is fixed, we use zeros for encoded columns
    for cat_col in encoder.get_feature_names_out(categorical_cols):
        user_df[cat_col] = 0
    # Scale numeric features
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

    # Predict Net Cash Flow
    prediction = xgb_model.predict(user_df[X.columns])[0]

    st.subheader("Calculated Cash Flow Components")
    st.write(f"Operating Cash Flow: {op_cf:.2f}")
    st.write(f"Investing Cash Flow: {inv_cf:.2f}")
    st.write(f"Financing Cash Flow: {fin_cf:.2f}")
    st.write(f"Sum of Calculated Cash Flows: {net_cf_calc:.2f}")

    st.subheader("Predicted Net Cash Flow Using Model")
    st.write(f"Predicted Net Cash Flow: {prediction:.2f}")
