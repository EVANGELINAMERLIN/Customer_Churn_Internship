Task Title  : Customer Behavior Analytics & Churn Prediction Dashboard
Task ID     : DA-INT-1
Domain      : Data Analytics

Description:
This project analyzes Telco customer data to identify churn patterns
and predict churn probability using Machine Learning models
(Logistic Regression and Random Forest).

Key Steps Performed:
1. Exploratory Data Analysis (EDA)
2. Data Cleaning (handled missing values in TotalCharges)
3. Feature Engineering (Service_Count, Avg_Monthly_Spend, 
   Is_LongTerm, Is_Monthly_Contract, Is_AutoPay)
4. Data Visualization (churn distribution, contract-wise churn,
   tenure analysis, correlation heatmap, customer segmentation)
5. ML Model Training & Evaluation (Logistic Regression, Random Forest)
   - Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
6. Customer Segmentation (High/Medium/Low value based on spending)
7. Churn Risk Prediction (Low/Medium/High risk categories)
8. Business Insights Report (top churn factors, high-risk traits,
   revenue impact estimation)

Bonus Features Implemented:
- Streamlit Dashboard (dashboard.py) - interactive web app for
  exploring churn data, customer segments, and live churn prediction
- SHAP Explainability - model interpretability showing why each
  customer was predicted to churn (global and individual explanations)
- Automated Weekly Prediction Pipeline (weekly_pipeline.py) - 
  automatically loads data, retrains model, predicts churn for all
  customers, and saves a timestamped report
- Email Report Generation - automatically emails a summary report
  with the CSV attachment after each pipeline run

How to Run:

1. Main Analysis Notebook:
   - Open Jupyter Notebook
   - Open Customer_Churn_Analysis.ipynb
   - Make sure the CSV file is in the same folder
   - Click Kernel > Restart & Run All

2. Streamlit Dashboard:
   - Open terminal in project folder
   - Run: streamlit run dashboard.py

3. Weekly Pipeline (with Email Report):
   - Open terminal in project folder
   - Run: python weekly_pipeline.py
   - Report saved in weekly_reports/ folder and emailed automatically

Files Included:
- Customer_Churn_Analysis.ipynb
- WA_Fn-UseC_-Telco-Customer-Churn.csv
- dashboard.py
- weekly_pipeline.py
- weekly_reports/ (sample generated reports)
