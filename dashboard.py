import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─── Title ─────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**TEYZIX CORE Internship | Task DA-INT-1**")
st.markdown("---")

# ─── Load & Prepare Data ───────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['Avg_Monthly_Spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = (df[service_cols] != 'No').sum(axis=1)
    df['Is_LongTerm'] = (df['tenure'] >= 24).astype(int)
    df['Is_Monthly_Contract'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Is_AutoPay'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
    df['Customer_Segment'] = pd.cut(
        df['MonthlyCharges'],
        bins=[0, 35, 65, 120],
        labels=['Low Value', 'Medium Value', 'High Value']
    )
    return df

@st.cache_resource
def train_model(df):
    df_ml = df.copy()
    df_ml.drop(['customerID', 'Customer_Segment'], axis=1, inplace=True)
    le = LabelEncoder()
    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = le.fit_transform(df_ml[col])
    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X_test, y_test, X

df = load_data()
rf_model, X_test, y_test, X = train_model(df)

# ─── Sidebar ───────────────────────────────────────────
st.sidebar.title("🔧 Filters")
segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    options=['Low Value', 'Medium Value', 'High Value'],
    default=['Low Value', 'Medium Value', 'High Value']
)
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique().tolist(),
    default=df['Contract'].unique().tolist()
)

filtered_df = df[
    (df['Customer_Segment'].isin(segment_filter)) &
    (df['Contract'].isin(contract_filter))
]

# ─── KPI Cards ─────────────────────────────────────────
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total = len(filtered_df)
churned = filtered_df['Churn'].sum()
churn_rate = (churned / total) * 100
avg_monthly = filtered_df['MonthlyCharges'].mean()

col1.metric("Total Customers", f"{total:,}")
col2.metric("Churned Customers", f"{churned:,}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

st.markdown("---")

# ─── Charts Row 1 ──────────────────────────────────────
st.subheader("📊 Churn Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Churn Distribution**")
    fig, ax = plt.subplots(figsize=(5,3))
    filtered_df['Churn'].value_counts().plot(
        kind='bar', color=['#2ecc71','#e74c3c'], ax=ax
    )
    ax.set_xticklabels(['Stayed', 'Churned'], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("**Churn by Contract Type**")
    fig, ax = plt.subplots(figsize=(5,3))
    churn_contract = filtered_df.groupby('Contract')['Churn'].mean() * 100
    churn_contract.plot(kind='bar', color='#e74c3c', ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    st.pyplot(fig)
    plt.close()

# ─── Charts Row 2 ──────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Customer Segmentation**")
    fig, ax = plt.subplots(figsize=(5,3))
    filtered_df['Customer_Segment'].value_counts().plot(
        kind='pie', autopct='%1.1f%%',
        colors=['#e74c3c','#f39c12','#2ecc71'], ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close()

with col4:
    st.markdown("**Monthly Charges Distribution**")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(filtered_df['MonthlyCharges'], bins=30, color='#3498db', edgecolor='white')
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ─── Feature Importance ────────────────────────────────
st.subheader("🔍 Top Churn Factors")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x='Importance', y='Feature', data=feature_importance,
            hue='Feature', palette='Reds_r', legend=False, ax=ax)
st.pyplot(fig)
plt.close()

st.markdown("---")

# ─── Churn Predictor ───────────────────────────────────
st.subheader("🎯 Predict Churn for a Customer")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
with col2:
    contract = st.selectbox("Contract Type", 
                ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox("Internet Service",
                ['DSL', 'Fiber optic', 'No'])
with col3:
    payment = st.selectbox("Payment Method",
                ['Electronic check', 'Mailed check',
                 'Bank transfer (automatic)', 'Credit card (automatic)'])
    senior = st.selectbox("Senior Citizen", ['No', 'Yes'])

if st.button("🔮 Predict Churn"):
    # Build sample input matching training features
    sample = X_test.iloc[0:1].copy()
    sample['tenure'] = tenure
    sample['MonthlyCharges'] = monthly_charges
    prob = rf_model.predict_proba(sample)[0][1]

    if prob >= 0.7:
        risk = "🔴 High Risk"
        color = "red"
    elif prob >= 0.4:
        risk = "🟠 Medium Risk"
        color = "orange"
    else:
        risk = "🟢 Low Risk"
        color = "green"

    st.markdown(f"### Churn Probability: **{prob*100:.1f}%**")
    st.markdown(f"### Risk Category: :{color}[{risk}]")

st.markdown("---")

# ─── Model Performance ─────────────────────────────────
st.subheader("🤖 Model Performance")
rf_pred = rf_model.predict(X_test)
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, rf_pred)*100:.2f}%")
col2.metric("F1 Score", f"{f1_score(y_test, rf_pred):.4f}")
col3.metric("ROC-AUC", f"{roc_auc_score(y_test, rf_pred):.4f}")

st.markdown("---")
st.markdown("**Built by Evangelina Merlin J | TEYZIX CORE Internship | Task DA-INT-1**")