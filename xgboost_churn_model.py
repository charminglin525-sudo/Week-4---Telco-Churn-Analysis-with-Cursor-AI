import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Define McKinsey-style color palette
COLORS = {
    'blue': ['#003057', '#005587', '#6BAED6'],
    'grey': ['#4B4B4B', '#9B9B9B', '#E5E5E5'],
    'accent': ['#1C6E8C', '#4CAF50', '#F28E2B', '#C72C41']
}

# Set default colors for plots
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [COLORS['blue'][0], COLORS['blue'][1], COLORS['accent'][0]])

print("="*80)
print("XGBOOST CHURN PREDICTION MODEL & RETENTION STRATEGY")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
print("1. DATA PREPROCESSING")
print("-"*80)

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Initial dataset shape: {df.shape}")

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Missing values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# Fill missing TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Handle zero TotalCharges (new customers)
df['TotalCharges'] = df['TotalCharges'].replace(0, df['TotalCharges'].median())

# Create target variable
df['Churn_binary'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Create grouping features in original df for later analysis
df['TenureGroup'] = pd.cut(df['tenure'], 
                           bins=[0, 12, 24, 36, 48, 60, 100],
                           labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+'])

df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'],
                                    bins=[0, 30, 50, 70, 90, 200],
                                    labels=['0-30', '30-50', '50-70', '70-90', '90+'])

df['RiskSegment'] = 'Low'
df.loc[(df['Contract'] == 'Month-to-month') & (df['tenure'] < 12), 'RiskSegment'] = 'High'
df.loc[(df['Contract'] == 'Month-to-month') & (df['tenure'] >= 12), 'RiskSegment'] = 'Medium'

# Create service count feature in original df
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['ServiceCount'] = df[service_cols].apply(
    lambda x: sum([1 for val in x if val not in ['No', 'No internet service']]), axis=1
)

# Drop customerID (not useful for prediction)
df_model = df.drop(['customerID', 'Churn'], axis=1)

print(f"Final dataset shape: {df_model.shape}")
print("✓ Data preprocessing completed\n")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("2. FEATURE ENGINEERING")
print("-"*80)

# Create new features
df_model['TenureGroup'] = pd.cut(df_model['tenure'], 
                                  bins=[0, 12, 24, 36, 48, 60, 100],
                                  labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+'])

df_model['MonthlyChargesGroup'] = pd.cut(df_model['MonthlyCharges'],
                                          bins=[0, 30, 50, 70, 90, 200],
                                          labels=['0-30', '30-50', '50-70', '70-90', '90+'])

df_model['TotalChargesGroup'] = pd.cut(df_model['TotalCharges'],
                                        bins=[0, 1000, 2000, 3000, 4000, 10000],
                                        labels=['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K+'])

# Create service count feature
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df_model['ServiceCount'] = df_model[service_cols].apply(
    lambda x: sum([1 for val in x if val not in ['No', 'No internet service']]), axis=1
)

# Create high-value customer flag
df_model['HighValue'] = ((df_model['MonthlyCharges'] > 70) & 
                         (df_model['Contract'] == 'Two year')).astype(int)

# Create risk segment
df_model['RiskSegment'] = 'Low'
df_model.loc[(df_model['Contract'] == 'Month-to-month') & 
             (df_model['tenure'] < 12), 'RiskSegment'] = 'High'
df_model.loc[(df_model['Contract'] == 'Month-to-month') & 
             (df_model['tenure'] >= 12), 'RiskSegment'] = 'Medium'

print("New features created:")
print("  - TenureGroup")
print("  - MonthlyChargesGroup")
print("  - TotalChargesGroup")
print("  - ServiceCount")
print("  - HighValue")
print("  - RiskSegment")
print("✓ Feature engineering completed\n")

# ============================================================================
# 3. DATA PREPARATION FOR MODELING
# ============================================================================
print("3. DATA PREPARATION FOR MODELING")
print("-"*80)

# Separate features and target
X = df_model.drop('Churn_binary', axis=1)
y = df_model['Churn_binary']

# Encode categorical variables (including category dtype)
label_encoders = {}
# Get both object and category type columns
categorical_cols = list(X.select_dtypes(include=['object']).columns) + \
                   list(X.select_dtypes(include=['category']).columns)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Churn rate in training: {y_train.mean():.2%}")
print(f"Churn rate in test: {y_test.mean():.2%}")
print("✓ Data preparation completed\n")

# ============================================================================
# 4. XGBOOST MODEL TRAINING
# ============================================================================
print("4. XGBOOST MODEL TRAINING")
print("-"*80)

# XGBoost parameters
params = {
    'n_estimators': 300,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'use_label_encoder': False
}

# Train model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=False)

print("✓ Model training completed\n")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("5. MODEL EVALUATION")
print("-"*80)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                Predicted")
print(f"              No      Yes")
print(f"Actual No   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Yes  {cm[1,0]:5d}   {cm[1,1]:5d}\n")

# Create Confusion Matrix visualization with McKinsey colors
from matplotlib.colors import LinearSegmentedColormap
colors_list = [COLORS['grey'][2], COLORS['blue'][0]]
n_bins = 100
mckinsey_cmap = LinearSegmentedColormap.from_list('mckinsey_blue', colors_list, N=n_bins)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=mckinsey_cmap, cbar_kws={'label': 'Count'}, 
            linewidths=1, linecolor='white', ax=ax,
            annot_kws={'color': 'white', 'fontweight': 'bold', 'fontsize': 12},
            vmin=0, vmax=cm.max())
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', 
             color=COLORS['grey'][0], pad=20)
ax.set_xlabel('Predicted', fontsize=11, color=COLORS['grey'][0])
ax.set_ylabel('Actual', fontsize=11, color=COLORS['grey'][0])
ax.set_xticklabels(['No', 'Yes'])
ax.set_yticklabels(['No', 'Yes'])
ax.tick_params(colors=COLORS['grey'][0], labelsize=10)
plt.tight_layout()
plt.savefig('Confusion_Matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Confusion Matrix saved to: Confusion_Matrix.png\n")

# Create ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=COLORS['blue'][0], linewidth=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color=COLORS['grey'][1], linestyle='--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['blue'][2])
ax.set_xlabel('False Positive Rate', fontsize=11, color=COLORS['grey'][0])
ax.set_ylabel('True Positive Rate', fontsize=11, color=COLORS['grey'][0])
ax.set_title('ROC Curve', fontsize=14, fontweight='bold', color=COLORS['grey'][0], pad=20)
ax.legend(frameon=False, fontsize=10, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(COLORS['grey'][2])
ax.spines['bottom'].set_color(COLORS['grey'][2])
ax.grid(axis='both', color=COLORS['grey'][2], linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(colors=COLORS['grey'][0], labelsize=10)
plt.tight_layout()
plt.savefig('ROC_Curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ ROC Curve saved to: ROC_Curve.png\n")

# Save evaluation metrics
evaluation_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Value': [accuracy, precision, recall, f1, auc]
}
eval_df = pd.DataFrame(evaluation_results)

# ============================================================================
# 6. SHAP FEATURE IMPORTANCE
# ============================================================================
print("6. SHAP FEATURE IMPORTANCE ANALYSIS")
print("-"*80)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:1000])  # Use subset for speed

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))
print("\n✓ SHAP analysis completed\n")

# ============================================================================
# 7. SEGMENT-BASED CHURN INSIGHTS
# ============================================================================
print("7. SEGMENT-BASED CHURN INSIGHTS")
print("-"*80)

# Add predictions to original dataframe
df['Churn_Probability'] = model.predict_proba(X)[:, 1]
df['Churn_Prediction'] = model.predict(X)
df['Churn_Risk'] = pd.cut(df['Churn_Probability'], 
                          bins=[0, 0.3, 0.6, 1.0],
                          labels=['Low', 'Medium', 'High'])

# Segment analysis
segments = []

# By Tenure
tenure_analysis = df.groupby('TenureGroup').agg({
    'Churn': lambda x: (x == 'Yes').sum() / len(x),
    'Churn_Probability': 'mean',
    'MonthlyCharges': 'mean',
    'customerID': 'count'
}).round(4)
tenure_analysis.columns = ['Actual_Churn_Rate', 'Avg_Churn_Probability', 'Avg_Monthly_Charges', 'Customer_Count']
segments.append(('Tenure', tenure_analysis))

# By Contract
contract_analysis = df.groupby('Contract').agg({
    'Churn': lambda x: (x == 'Yes').sum() / len(x),
    'Churn_Probability': 'mean',
    'MonthlyCharges': 'mean',
    'customerID': 'count'
}).round(4)
contract_analysis.columns = ['Actual_Churn_Rate', 'Avg_Churn_Probability', 'Avg_Monthly_Charges', 'Customer_Count']
segments.append(('Contract', contract_analysis))

# By Monthly Charges
charges_analysis = df.groupby('MonthlyChargesGroup').agg({
    'Churn': lambda x: (x == 'Yes').sum() / len(x),
    'Churn_Probability': 'mean',
    'MonthlyCharges': 'mean',
    'customerID': 'count'
}).round(4)
charges_analysis.columns = ['Actual_Churn_Rate', 'Avg_Churn_Probability', 'Avg_Monthly_Charges', 'Customer_Count']
segments.append(('MonthlyCharges', charges_analysis))

# By Internet Service
internet_analysis = df.groupby('InternetService').agg({
    'Churn': lambda x: (x == 'Yes').sum() / len(x),
    'Churn_Probability': 'mean',
    'MonthlyCharges': 'mean',
    'customerID': 'count'
}).round(4)
internet_analysis.columns = ['Actual_Churn_Rate', 'Avg_Churn_Probability', 'Avg_Monthly_Charges', 'Customer_Count']
segments.append(('InternetService', internet_analysis))

# By Risk Segment
risk_analysis = df.groupby('RiskSegment').agg({
    'Churn': lambda x: (x == 'Yes').sum() / len(x),
    'Churn_Probability': 'mean',
    'MonthlyCharges': 'mean',
    'customerID': 'count'
}).round(4)
risk_analysis.columns = ['Actual_Churn_Rate', 'Avg_Churn_Probability', 'Avg_Monthly_Charges', 'Customer_Count']
segments.append(('RiskSegment', risk_analysis))

print("✓ Segment analysis completed\n")

# ============================================================================
# 8. RETENTION STRATEGY BY SEGMENT
# ============================================================================
print("8. RETENTION STRATEGY DEVELOPMENT")
print("-"*80)

# Identify high-risk customers
high_risk_customers = df[df['Churn_Risk'] == 'High'].copy()
high_risk_customers = high_risk_customers.sort_values('Churn_Probability', ascending=False)

# Calculate potential revenue at risk
high_risk_customers['Monthly_Revenue_At_Risk'] = high_risk_customers['MonthlyCharges']
total_revenue_at_risk = high_risk_customers['Monthly_Revenue_At_Risk'].sum()

print(f"High-risk customers: {len(high_risk_customers):,}")
print(f"Total monthly revenue at risk: ${total_revenue_at_risk:,.2f}\n")

# Create retention plan
retention_plan = []

# Segment 1: Month-to-month, Low Tenure
segment1 = high_risk_customers[
    (high_risk_customers['Contract'] == 'Month-to-month') & 
    (high_risk_customers['tenure'] < 12)
]
if len(segment1) > 0:
    retention_plan.append({
        'Segment': 'Month-to-Month, Low Tenure (<12 months)',
        'Customer_Count': len(segment1),
        'Avg_Churn_Probability': float(segment1['Churn_Probability'].mean()),
        'Monthly_Revenue_At_Risk': float(segment1['MonthlyCharges'].sum()),
        'Strategy': 'Offer 1-year contract with 10-15% discount',
        'Action_Items': [
            'Proactive outreach within first 3 months',
            'Offer loyalty discount for contract conversion',
            'Assign dedicated retention specialist',
            'Provide onboarding support package'
        ],
        'Expected_Impact': 'Reduce churn by 30-40%',
        'Cost_per_Customer': '$15-20/month discount',
        'ROI': 'High - Early intervention prevents long-term loss'
    })

# Segment 2: High Monthly Charges, Month-to-Month
segment2 = high_risk_customers[
    (high_risk_customers['MonthlyCharges'] > 70) & 
    (high_risk_customers['Contract'] == 'Month-to-month')
]
if len(segment2) > 0:
    retention_plan.append({
        'Segment': 'High Value, Month-to-Month',
        'Customer_Count': len(segment2),
        'Avg_Churn_Probability': float(segment2['Churn_Probability'].mean()),
        'Monthly_Revenue_At_Risk': float(segment2['MonthlyCharges'].sum()),
        'Strategy': 'Convert to annual contract with premium benefits',
        'Action_Items': [
            'Offer 2-year contract with 20% discount',
            'Provide premium service add-ons',
            'Priority customer service access',
            'Exclusive loyalty rewards program'
        ],
        'Expected_Impact': 'Reduce churn by 50-60%',
        'Cost_per_Customer': '$20-30/month discount',
        'ROI': 'Very High - High-value customers worth retaining'
    })

# Segment 3: Fiber Optic, High Churn Risk
segment3 = high_risk_customers[high_risk_customers['InternetService'] == 'Fiber optic']
if len(segment3) > 0:
    retention_plan.append({
        'Segment': 'Fiber Optic, High Churn Risk',
        'Customer_Count': len(segment3),
        'Avg_Churn_Probability': float(segment3['Churn_Probability'].mean()),
        'Monthly_Revenue_At_Risk': float(segment3['MonthlyCharges'].sum()),
        'Strategy': 'Improve service quality and add value',
        'Action_Items': [
            'Service quality audit and improvement',
            'Offer speed upgrade or price match',
            'Add complementary services (security, backup)',
            'Proactive technical support'
        ],
        'Expected_Impact': 'Reduce churn by 25-35%',
        'Cost_per_Customer': '$10-15/month in service improvements',
        'ROI': 'Medium-High - Address root cause of dissatisfaction'
    })

# Segment 4: Low Service Count, High Risk
segment4 = high_risk_customers[high_risk_customers['ServiceCount'] < 2]
if len(segment4) > 0:
    retention_plan.append({
        'Segment': 'Low Service Engagement, High Risk',
        'Customer_Count': len(segment4),
        'Avg_Churn_Probability': float(segment4['Churn_Probability'].mean()),
        'Monthly_Revenue_At_Risk': float(segment4['MonthlyCharges'].sum()),
        'Strategy': 'Increase service stickiness',
        'Action_Items': [
            'Offer bundled services at discounted rate',
            'Free trial of premium services (Security, Backup)',
            'Educational content on service benefits',
            'Referral program incentives'
        ],
        'Expected_Impact': 'Reduce churn by 20-30%',
        'Cost_per_Customer': '$5-10/month in bundled discounts',
        'ROI': 'Medium - Increases customer lifetime value'
    })

# Segment 5: Streaming Services Users, High Risk
segment5 = high_risk_customers[
    (high_risk_customers['StreamingTV'] == 'Yes') | 
    (high_risk_customers['StreamingMovies'] == 'Yes')
]
if len(segment5) > 0:
    retention_plan.append({
        'Segment': 'Streaming Services Users, High Risk',
        'Customer_Count': len(segment5),
        'Avg_Churn_Probability': float(segment5['Churn_Probability'].mean()),
        'Monthly_Revenue_At_Risk': float(segment5['MonthlyCharges'].sum()),
        'Strategy': 'Enhance streaming experience',
        'Action_Items': [
            'Improve streaming quality and reliability',
            'Offer exclusive content or partnerships',
            'Bundle streaming with other services',
            'Provide streaming device discounts'
        ],
        'Expected_Impact': 'Reduce churn by 15-25%',
        'Cost_per_Customer': '$8-12/month in improvements',
        'ROI': 'Medium - Addresses specific pain point'
    })

# Convert Action_Items list to string for DataFrame compatibility
retention_plan_for_df = []
for plan in retention_plan:
    plan_copy = plan.copy()
    plan_copy['Action_Items'] = '; '.join(plan['Action_Items'])  # Convert list to string
    # Ensure numeric values are properly typed
    plan_copy['Customer_Count'] = int(plan_copy['Customer_Count'])
    plan_copy['Avg_Churn_Probability'] = float(plan_copy['Avg_Churn_Probability'])
    plan_copy['Monthly_Revenue_At_Risk'] = float(plan_copy['Monthly_Revenue_At_Risk'])
    retention_plan_for_df.append(plan_copy)

retention_plan_df = pd.DataFrame(retention_plan_for_df)
print("✓ Retention strategy developed\n")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("9. SAVING RESULTS")
print("-"*80)

# Create Excel writer
with pd.ExcelWriter('XGBoost_Churn_Model_Results.xlsx', engine='openpyxl') as writer:
    # Model Evaluation
    eval_df.to_excel(writer, sheet_name='Model_Evaluation', index=False)
    
    # Feature Importance
    feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
    
    # Segment Analysis
    for name, analysis in segments:
        analysis.to_excel(writer, sheet_name=f'Segment_{name}', index=True)
    
    # Retention Plan
    retention_plan_df.to_excel(writer, sheet_name='Retention_Plan', index=False)
    
    # High Risk Customers (sample)
    # Get all feature columns used in the model (from df_model, excluding Churn_binary)
    model_feature_columns = [col for col in df_model.columns if col != 'Churn_binary']
    
    # Start with basic information columns (A-H)
    basic_columns = ['customerID', 'Churn_Probability', 'Churn_Risk', 
                     'Contract', 'tenure', 'MonthlyCharges',
                     'InternetService', 'Monthly_Revenue_At_Risk']
    
    # Get all model feature columns from df_model (starting from I column)
    # Both df and df_model share the same index, so we can directly merge by index
    high_risk_indices = high_risk_customers.index
    model_features_for_high_risk = df_model.loc[high_risk_indices, model_feature_columns].copy()
    
    # Combine basic info with model features (by index)
    high_risk_sample = high_risk_customers[basic_columns].copy()
    high_risk_sample = pd.concat([high_risk_sample, model_features_for_high_risk], axis=1)
    high_risk_sample = high_risk_sample.head(1000)
    
    high_risk_sample.to_excel(writer, sheet_name='High_Risk_Customers', index=False)

print("✓ Results saved to: XGBoost_Churn_Model_Results.xlsx")

# Save detailed report to text file
with open('XGBoost_Churn_Model_Report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("XGBOOST CHURN PREDICTION MODEL & RETENTION STRATEGY REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("1. MODEL EVALUATION METRICS\n")
    f.write("="*80 + "\n")
    f.write(eval_df.to_string(index=False) + "\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("2. TOP 15 FEATURE IMPORTANCE\n")
    f.write("="*80 + "\n")
    f.write(feature_importance.head(15).to_string(index=False) + "\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("3. SEGMENT-BASED CHURN ANALYSIS\n")
    f.write("="*80 + "\n")
    for name, analysis in segments:
        f.write(f"\n{name}:\n")
        f.write("-"*80 + "\n")
        f.write(analysis.to_string() + "\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("4. RETENTION STRATEGY BY SEGMENT\n")
    f.write("="*80 + "\n\n")
    
    for idx, plan in enumerate(retention_plan, 1):
        f.write(f"\nSegment {idx}: {plan['Segment']}\n")
        f.write("-"*80 + "\n")
        f.write(f"Customer Count: {plan['Customer_Count']:,}\n")
        f.write(f"Average Churn Probability: {plan['Avg_Churn_Probability']:.2%}\n")
        f.write(f"Monthly Revenue At Risk: ${plan['Monthly_Revenue_At_Risk']:,.2f}\n")
        f.write(f"\nStrategy: {plan['Strategy']}\n")
        f.write(f"\nAction Items:\n")
        for item in plan['Action_Items']:
            f.write(f"  • {item}\n")
        f.write(f"\nExpected Impact: {plan['Expected_Impact']}\n")
        f.write(f"Cost per Customer: {plan['Cost_per_Customer']}\n")
        f.write(f"ROI: {plan['ROI']}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("5. SUMMARY & RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total High-Risk Customers: {len(high_risk_customers):,}\n")
    f.write(f"Total Monthly Revenue At Risk: ${total_revenue_at_risk:,.2f}\n")
    f.write(f"Annual Revenue At Risk: ${total_revenue_at_risk * 12:,.2f}\n\n")
    
    f.write("Key Recommendations:\n")
    f.write("1. Prioritize Month-to-Month customers with low tenure for immediate intervention\n")
    f.write("2. Focus on high-value customers (>$70/month) for contract conversion\n")
    f.write("3. Address service quality issues for Fiber Optic customers\n")
    f.write("4. Increase service engagement for customers with low service count\n")
    f.write("5. Implement proactive retention campaigns based on churn probability thresholds\n")
    f.write("6. Monitor model performance monthly and retrain quarterly\n\n")

print("✓ Report saved to: XGBoost_Churn_Model_Report.txt")

# Create SHAP summary plot with McKinsey colors
try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:1000], plot_type="bar", show=False, 
                     color=COLORS['blue'][0])
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', 
              color=COLORS['grey'][0], pad=20)
    plt.xlabel('Mean(|SHAP value|)', fontsize=11, color=COLORS['grey'][0])
    plt.tick_params(colors=COLORS['grey'][0], labelsize=10)
    plt.tight_layout()
    plt.savefig('SHAP_Feature_Importance.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ SHAP plot saved to: SHAP_Feature_Importance.png")
except:
    # Fallback if SHAP plot fails
    print("⚠ SHAP plot generation skipped (using alternative visualization)")

# Create Feature Importance bar chart
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
               color=COLORS['blue'][0], alpha=0.85)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.set_xlabel('Feature Importance', fontsize=11, color=COLORS['grey'][0])
ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold', 
             color=COLORS['grey'][0], pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(COLORS['grey'][2])
ax.spines['bottom'].set_color(COLORS['grey'][2])
ax.grid(axis='x', color=COLORS['grey'][2], linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(colors=COLORS['grey'][0], labelsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Feature_Importance_Bar.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Feature Importance bar chart saved to: Feature_Importance_Bar.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  - XGBoost_Churn_Model_Results.xlsx")
print("  - XGBoost_Churn_Model_Report.txt")
print("  - Confusion_Matrix.png")
print("  - ROC_Curve.png")
print("  - SHAP_Feature_Importance.png")
print("  - Feature_Importance_Bar.png")

