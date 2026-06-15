"""
TEYZIX CORE Internship - Task DA-INT-1
Automated Weekly Churn Prediction Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("="*60)
print(f"🚀 STARTING WEEKLY CHURN PREDICTION PIPELINE")
print(f"📅 Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ─── STEP 1: Load Data ─────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"     ✅ Loaded {len(df)} customer records")

# ─── STEP 2: Clean Data ────────────────────────────────
print("\n[2/6] Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("     ✅ Missing values handled")

# ─── STEP 3: Feature Engineering ───────────────────────
print("\n[3/6] Creating features...")
df['Avg_Monthly_Spend'] = df['TotalCharges'] / (df['tenure'] + 1)
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Service_Count'] = (df[service_cols] != 'No').sum(axis=1)
df['Is_LongTerm'] = (df['tenure'] >= 24).astype(int)
df['Is_Monthly_Contract'] = (df['Contract'] == 'Month-to-month').astype(int)
df['Is_AutoPay'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
df['Customer_Segment'] = pd.cut(
    df['MonthlyCharges'], bins=[0, 35, 65, 120],
    labels=['Low Value', 'Medium Value', 'High Value']
)
print("     ✅ Behavioral features created")

# ─── STEP 4: Train Model ───────────────────────────────
print("\n[4/6] Training model...")
df_ml = df.copy()
customer_ids = df_ml['customerID']
df_ml.drop(['customerID', 'Customer_Segment'], axis=1, inplace=True)

le = LabelEncoder()
for col in df_ml.select_dtypes(include='object').columns:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print(f"     ✅ Model trained on {len(X_train)} customers")

# ─── STEP 5: Predict for All Customers ─────────────────
print("\n[5/6] Generating predictions for all customers...")
all_proba = rf_model.predict_proba(X)[:, 1]

def assign_risk(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

results = pd.DataFrame({
    'CustomerID': customer_ids,
    'Churn_Probability': all_proba.round(4),
    'Risk_Category': [assign_risk(p) for p in all_proba],
    'MonthlyCharges': df['MonthlyCharges'],
    'Contract': df['Contract'],
    'Tenure': df['tenure']
})

results = results.sort_values('Churn_Probability', ascending=False)
print(f"     ✅ Predictions generated for {len(results)} customers")

# ─── STEP 6: Save Report ───────────────────────────────
print("\n[6/6] Saving report...")

os.makedirs("weekly_reports", exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"weekly_reports/churn_report_{timestamp}.csv"
results.to_csv(filename, index=False)

print(f"     ✅ Report saved as: {filename}")

# ─── SUMMARY ────────────────────────────────────────────
print("\n" + "="*60)
print("📊 PIPELINE SUMMARY")
print("="*60)
print(f"Total Customers Analyzed : {len(results)}")
print(f"\nRisk Distribution:")
print(results['Risk_Category'].value_counts().to_string())
print(f"\nTop 5 Highest Risk Customers:")
print(results.head(5)[['CustomerID', 'Churn_Probability', 'Risk_Category']].to_string(index=False))

high_risk_revenue = results[results['Risk_Category']=='High Risk']['MonthlyCharges'].sum()
print(f"\n💰 Monthly Revenue at High Risk: ${high_risk_revenue:,.2f}")

print("\n" + "="*60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
# ─── EMAIL REPORT ───────────────────────────────────────
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

print("\nSending email report...")

# --- EDIT THESE 3 LINES ---
SENDER_EMAIL = "jevangelinamerlin@gmail.com"        # your Gmail address
APP_PASSWORD = "your_app_password_here" 
      # the App Password (no spaces)
RECEIVER_EMAIL = "jevangelinamerlin@gmail.com"      # who receives the report (can be same email)
# ---------------------------

subject = "Weekly Churn Prediction Report - " + datetime.now().strftime('%Y-%m-%d')

body = f"""
Hello,

The weekly churn prediction pipeline has completed successfully.

Summary:
- Total Customers Analyzed: {len(results)}
- High Risk Customers: {(results['Risk_Category']=='High Risk').sum()}
- Medium Risk Customers: {(results['Risk_Category']=='Medium Risk').sum()}
- Low Risk Customers: {(results['Risk_Category']=='Low Risk').sum()}
- Monthly Revenue at High Risk: ${high_risk_revenue:,.2f}

The full report is attached.

Regards,
Automated Churn Prediction Pipeline
TEYZIX CORE Internship
"""

msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

# Attach the CSV report
with open(filename, 'rb') as f:
    attachment = MIMEApplication(f.read(), _subtype='csv')
    attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(filename))
    msg.attach(attachment)

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.send_message(msg)
    server.quit()
    print("Email sent successfully to", RECEIVER_EMAIL)
except Exception as e:
    print("Email sending failed:", e)