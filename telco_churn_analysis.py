import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [COLORS['blue'][0], COLORS['blue'][1], COLORS['accent'][0])

print("="*80)
print("TELCO CUSTOMER CHURN ANALYSIS - DESCRIPTIVE ANALYTICS")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Dataset shape: {df.shape}\n")

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['TotalCharges'] = df['TotalCharges'].replace(0, df['TotalCharges'].median())

# Create output directories
output_dirs = [
    'visualization/01. Customer Churn Overview',
    'visualization/02. Customer tenure',
    'visualization/03. Service Product Impact'
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

print("✓ Output directories created\n")

# ============================================================================
# HELPER FUNCTION: Apply McKinsey style to plots
# ============================================================================
def apply_mckinsey_style(ax, title, xlabel='', ylabel='', legend=True):
    """Apply McKinsey-style formatting to matplotlib axes"""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grey'][2])
    ax.spines['bottom'].set_color(COLORS['grey'][2])
    ax.grid(axis='both', color=COLORS['grey'][2], linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['grey'][0], pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=COLORS['grey'][0])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=COLORS['grey'][0])
    
    ax.tick_params(colors=COLORS['grey'][0], labelsize=10)
    
    if legend and ax.get_legend():
        ax.get_legend().set_frame_on(False)
        ax.get_legend().set_fontsize(10)

# ============================================================================
# 1. CUSTOMER CHURN OVERVIEW
# ============================================================================
print("="*80)
print("1. CUSTOMER CHURN OVERVIEW")
print("="*80)

# 1.1 Customer Count Pie Chart
churn_counts = df['Churn'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
colors_pie = [COLORS['grey'][1], COLORS['accent'][3]]  # Grey for Yes, Red for No
ax.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
       colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold', 
             color=COLORS['grey'][0], pad=20)
plt.tight_layout()
plt.savefig('visualization/01. Customer Churn Overview/churn_customer_count_pie.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: churn_customer_count_pie.png")

# 1.2 Revenue Pie Chart
churn_revenue = df.groupby('Churn')['MonthlyCharges'].sum()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(churn_revenue.values, labels=churn_revenue.index, autopct='%1.1f%%',
       colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Revenue Distribution by Churn Status', fontsize=14, fontweight='bold',
             color=COLORS['grey'][0], pad=20)
plt.tight_layout()
plt.savefig('visualization/01. Customer Churn Overview/churn_revenue_pie.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: churn_revenue_pie.png")

# 1.3 Monthly Charges Bar Chart (Count)
charges_churn = df.groupby(['MonthlyCharges', 'Churn']).size().unstack(fill_value=0)
charges_churn.plot(kind='bar', stacked=True, color=[COLORS['grey'][1], COLORS['accent'][3]], 
                   ax=plt.gca(), width=0.8)
apply_mckinsey_style(plt.gca(), 'Monthly Charges by Churn Status (Count)', 
                     'Monthly Charges', 'Customer Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualization/01. Customer Churn Overview/charges_churn_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: charges_churn_bar.png")

# 1.4 Monthly Charges Bar Chart (Percentage)
charges_pct = df.groupby('MonthlyCharges')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(charges_pct.index.astype(str), charges_pct.values, 
              color=COLORS['accent'][3], alpha=0.85)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
apply_mckinsey_style(ax, 'Monthly Charges Churn Rate (%)', 'Monthly Charges', 'Churn Rate (%)', legend=False)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualization/01. Customer Churn Overview/charges_churn_pct_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: charges_churn_pct_bar.png\n")

# ============================================================================
# 2. CUSTOMER TENURE ANALYSIS
# ============================================================================
print("="*80)
print("2. CUSTOMER TENURE ANALYSIS")
print("="*80)

# 2.1 Create Tenure Groups
df['TenureGroup'] = pd.cut(df['tenure'], 
                           bins=[0, 12, 24, 36, 48, 60, 100],
                           labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+'])

# 2.2 Tenure Churn Bar Chart (Count)
tenure_churn = df.groupby(['TenureGroup', 'Churn']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
tenure_churn.plot(kind='bar', stacked=True, color=[COLORS['grey'][1], COLORS['accent'][3]], 
                  ax=ax, width=0.8)
apply_mckinsey_style(ax, 'Churn by Tenure Group (Count)', 'Tenure (months)', 'Customer Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualization/02. Customer tenure/tenure_churn_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: tenure_churn_bar.png")

# 2.3 Tenure Churn Bar Chart (Percentage)
tenure_pct = df.groupby('TenureGroup')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(tenure_pct.index.astype(str), tenure_pct.values,
              color=COLORS['accent'][3], alpha=0.85)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
apply_mckinsey_style(ax, 'Churn Rate by Tenure Group (%)', 'Tenure (months)', 'Churn Rate (%)', legend=False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualization/02. Customer tenure/tenure_churn_pct_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: tenure_churn_pct_bar.png")

# 2.4 Contract Churn Bar Chart (Count)
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
contract_churn.plot(kind='bar', stacked=True, color=[COLORS['grey'][1], COLORS['accent'][3]], 
                    ax=ax, width=0.8)
apply_mckinsey_style(ax, 'Churn by Contract Type (Count)', 'Contract Type', 'Customer Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualization/02. Customer tenure/contract_churn_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: contract_churn_bar.png")

# 2.5 Contract Churn Bar Chart (Percentage)
contract_pct = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(contract_pct.index, contract_pct.values,
              color=COLORS['accent'][3], alpha=0.85)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
apply_mckinsey_style(ax, 'Churn Rate by Contract Type (%)', 'Contract Type', 'Churn Rate (%)', legend=False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualization/02. Customer tenure/contract_churn_pct_bar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: contract_churn_pct_bar.png\n")

# ============================================================================
# 3. SERVICE PRODUCT IMPACT
# ============================================================================
print("="*80)
print("3. SERVICE PRODUCT IMPACT")
print("="*80)

# 3.1 Internet Service Heatmap
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(internet_churn, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'},
            linewidths=1, linecolor='white', ax=ax, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('Internet Service Type vs Churn Rate (%)', fontsize=14, fontweight='bold',
             color=COLORS['grey'][0], pad=20)
ax.set_xlabel('Churn', fontsize=11, color=COLORS['grey'][0])
ax.set_ylabel('Internet Service', fontsize=11, color=COLORS['grey'][0])
ax.tick_params(colors=COLORS['grey'][0], labelsize=10)
plt.tight_layout()
plt.savefig('visualization/03. Service Product Impact/internet_service_heatmap.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: internet_service_heatmap.png")

# 3.2 Service Comparison (Percentage)
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

service_churn_pct = {}
for col in service_cols:
    service_df = df[df[col].isin(['Yes', 'No'])]  # Exclude 'No internet service'
    service_churn_pct[col] = service_df.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )

service_df_pct = pd.DataFrame(service_churn_pct)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(service_cols))
width = 0.35
bars1 = ax.bar(x - width/2, service_df_pct.loc['No', :], width, 
               label='No', color=COLORS['grey'][1], alpha=0.85)
bars2 = ax.bar(x + width/2, service_df_pct.loc['Yes', :], width,
               label='Yes', color=COLORS['blue'][0], alpha=0.85)
ax.set_xlabel('Service Type', fontsize=11, color=COLORS['grey'][0])
ax.set_ylabel('Churn Rate (%)', fontsize=11, color=COLORS['grey'][0])
ax.set_title('Churn Rate by Service Type (%)', fontsize=14, fontweight='bold',
             color=COLORS['grey'][0], pad=20)
ax.set_xticks(x)
ax.set_xticklabels([col.replace('Streaming', 'Stream.').replace('Online', 'Onl.') 
                    for col in service_cols], rotation=45, ha='right')
apply_mckinsey_style(ax, 'Churn Rate by Service Type (%)', '', 'Churn Rate (%)')
plt.tight_layout()
plt.savefig('visualization/03. Service Product Impact/services_churn_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: services_churn_comparison.png")

# 3.3 Service Comparison (Count)
service_churn_count = {}
for col in service_cols:
    service_df = df[df[col].isin(['Yes', 'No'])]
    service_churn_count[col] = service_df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').sum())

service_df_count = pd.DataFrame(service_churn_count)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(service_cols))
width = 0.35
bars1 = ax.bar(x - width/2, service_df_count.loc['No', :], width,
               label='No', color=COLORS['grey'][1], alpha=0.85)
bars2 = ax.bar(x + width/2, service_df_count.loc['Yes', :], width,
               label='Yes', color=COLORS['blue'][0], alpha=0.85)
ax.set_xlabel('Service Type', fontsize=11, color=COLORS['grey'][0])
ax.set_ylabel('Churn Count', fontsize=11, color=COLORS['grey'][0])
ax.set_xticks(x)
ax.set_xticklabels([col.replace('Streaming', 'Stream.').replace('Online', 'Onl.') 
                    for col in service_cols], rotation=45, ha='right')
apply_mckinsey_style(ax, 'Churn Count by Service Type', '', 'Churn Count')
plt.tight_layout()
plt.savefig('visualization/03. Service Product Impact/services_churn_comparison_count.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: services_churn_comparison_count.png")

# 3.4 Individual Service Charts
for col in service_cols:
    service_df = df[df[col].isin(['Yes', 'No'])]
    service_pct = service_df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_service = [COLORS['grey'][1] if idx == 'No' else COLORS['blue'][0] 
                      for idx in service_pct.index]
    bars = ax.bar(service_pct.index, service_pct.values, color=colors_service, alpha=0.85)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=11, fontweight='bold')
    apply_mckinsey_style(ax, f'{col} - Churn Rate (%)', col, 'Churn Rate (%)', legend=False)
    plt.tight_layout()
    filename = col.lower().replace('streaming', 'streaming').replace('online', 'online')
    plt.savefig(f'visualization/03. Service Product Impact/{filename}_churn.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {filename}_churn.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to 'visualization/' directory")
print(f"Total charts generated: {sum(len(os.listdir(d)) for d in output_dirs)}")

