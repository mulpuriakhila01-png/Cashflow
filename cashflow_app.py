# cashflow_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

st.set_page_config(page_title="Cash Flow Prediction App", layout="wide")
st.title("💰 Cash Flow Prediction using XGBoost")
st.markdown("Upload your financial dataset to predict **Net Cash Flow** using engineered features and XGBoost regression.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("### Dataset Shape:", data.shape)
    st.write("### Missing Values:")
    missing = data.isna().sum()
    st.dataframe(missing[missing > 0])

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
        data['Debt_Raised']
        - data['Debt_Repaid']
        - data['Interest_Paid']
        + data['Equity_Issued']
        - data['Dividends_Paid']
    )

    data['Cash_Flow'] = (
        data['Operating_Cash_Flow']
        + data['Investing_Cash_Flow']
        + data['Financing_Cash_Flow']
    )

    st.subheader("🧮 Engineered Features (First 5 rows)")
    st.dataframe(data[['Company_Name','Month','Operating_Cash_Flow','Investing_Cash_Flow','Financing_Cash_Flow','Cash_Flow']].head())

    engineered_features = [ 'Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']
    numeric_cols.extend(engineered_features)

    st.subheader("🔍 Correlation Heatmap")
    plt.figure(figsize=(12,7))
    sns.heatmap(data[numeric_cols + ['Net_Cash_Flow']].corr(), annot=False, cmap='coolwarm')
    st.pyplot(plt)

    def cap_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower, lower,
                              np.where(df[column] > upper, upper, df[column]))
        return df

    for col in numeric_cols:
        data = cap_outliers_iqr(data, col)

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

    st.subheader("🚀 Model Training: XGBoost Regression")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n_samples = len(y_test)
    n_features = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    st.write("### 📈 Model Evaluation Metrics")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**Adjusted R²:** {adj_r2:.4f}")

        # 🔮 Predict for entire dataset
    if st.button("🔮 Predict Net Cash Flow for Entire Dataset"):
        data_encoded['Predicted_Net_Cash_Flow'] = xgb_model.predict(X)

        # Reverse scaling for better readability (optional)
        data['Predicted_Net_Cash_Flow'] = data_encoded['Predicted_Net_Cash_Flow']

        st.subheader("📈 Predicted Results (First 10 Rows)")
        st.dataframe(
            data[['Company_Name', 'Month', 'Net_Cash_Flow', 'Predicted_Net_Cash_Flow']].head(10)
        )

        # 📊 Visualization: Actual vs Predicted for All
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.scatterplot(
            x=data['Net_Cash_Flow'], y=data['Predicted_Net_Cash_Flow'],
            color='blue', alpha=0.6
        )
        ax2.plot(
            [data['Net_Cash_Flow'].min(), data['Net_Cash_Flow'].max()],
            [data['Net_Cash_Flow'].min(), data['Net_Cash_Flow'].max()],
            color='red', linestyle='--'
        )
        plt.xlabel("Actual Net Cash Flow")
        plt.ylabel("Predicted Net Cash Flow")
        plt.title("Actual vs Predicted Net Cash Flow (All Data)")
        st.pyplot(fig2)

        # 💾 Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predicted Dataset as CSV",
            data=csv,
            file_name="Predicted_CashFlow.csv",
            mime="text/csv",
        )

    st.subheader("🔄 5-Fold Cross Validation")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring='r2')
    st.write(f"Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    st.subheader("📊 Actual vs Predicted Net Cash Flow")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=y_test, y=y_pred, color='green', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Net Cash Flow (XGBoost)")
    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to begin analysis and prediction.")
