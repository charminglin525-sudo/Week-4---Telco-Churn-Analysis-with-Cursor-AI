# Telco Customer Churn – Analytics & XGBoost Modeling

以麥肯錫風格完成 Telco 客戶流失分析與 XGBoost 預測模型。本專案包含雙語說明，方便商務與技術團隊協作。

---

## 📂 專案結構 Project Structure

- `telco_churn_analysis.py`  
  客戶流失描述型分析與麥肯錫風格視覺化 (Tenure / Charges / Contract / Services)  
- `xgboost_churn_model.py`  
  端對端 XGBoost 流失預測模型、SHAP 解釋、Retention Plan  
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
  原始資料集 (7,043 筆)  
- `visualization/`  
  所有分析圖表 (PNG)  
- `今日進度1126.txt`  
  2024-11-26 工作紀錄 (中英雙語)  
- `XGBoost_Churn_Model_Results.xlsx` *(run script to generate)*  
- `XGBoost_Churn_Model_Report.txt` *(run script to generate)*  
- `Confusion_Matrix.png`, `ROC_Curve.png`, `SHAP_Feature_Importance.png`, `Feature_Importance_Bar.png`

---

## 🛠 環境需求 Requirements

```bash
python >= 3.10
pip install -r requirements.txt  # 或手動安裝下列套件
```

必要套件 / Key packages:
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- xgboost
- shap
- openpyxl (輸出 Excel)

---

## 🚀 使用方式 How to Run

1. **資料分析與視覺化 (Descriptive Analytics)**
   ```bash
   python telco_churn_analysis.py
   ```
   輸出：`visualization/` 內所有圖表 (麥肯錫深藍配色、含百分比標註)。

2. **XGBoost 模型與 Retention Plan**
   ```bash
   python xgboost_churn_model.py
   ```
   輸出：
   - `XGBoost_Churn_Model_Results.xlsx`
   - `XGBoost_Churn_Model_Report.txt`
   - `Confusion_Matrix.png`, `ROC_Curve.png`, `SHAP_Feature_Importance.png`, `Feature_Importance_Bar.png`

> 建議先確認 `WA_Fn-UseC_-Telco-Customer-Churn.csv` 與腳本位於同一路徑。

---

## 📊 功能摘要 Highlights

### telco_churn_analysis.py
- 客戶流失概覽：Customer count & revenue pie charts
- Tenure / Monthly Charges / Contract 三大構面：數量＋百分比長條圖
- 服務產品影響：熱力圖＋六大服務比較圖 (百分比／客戶數量)
- 麥肯錫風格配色 (#003057 / #005587 / #6BAED6) 及圖表調教 (無框 legend、百分比標籤等)

### xgboost_churn_model.py
- Data preprocessing + Feature engineering (TenureGroup, MonthlyChargesGroup, ServiceCount, RiskSegment…)
- XGBoost 訓練、AUC/Precision/Recall/F1/Confusion Matrix/ROC
- SHAP feature importance + Top 15 feature bar chart
- Segment-based churn insights (Tenure / Contract / ARPU / Internet type / Risk segment)
- 五大 Retention Plan：含客戶數、流失機率、Revenue at Risk、策略與 ROI 評估

---

## 📁 主要輸出檔案 Outputs

| 檔案 | 內容 |
|------|------|
| `visualization/*.png` | 全部描述型圖表 |
| `Confusion_Matrix.png` | 麥肯錫藍色系混淆矩陣 |
| `ROC_Curve.png` | 模型 ROC 曲線 |
| `SHAP_Feature_Importance.png` | SHAP 柱狀圖 |
| `Feature_Importance_Bar.png` | XGBoost 重要特徵 Top 15 |
| `XGBoost_Churn_Model_Results.xlsx` | 評估指標、特徵重要性、各 Segment 洞察、Retention Plan、High-Risk 客戶清單 |
| `XGBoost_Churn_Model_Report.txt` | 詳細文字報告 (含策略建議) |
| `今日進度1126.txt` | 2024/11/26 進度紀錄 (中英文) |

---

## ✅ 待辦 / Next Steps

- 進一步最佳化 XGBoost 參數與交叉驗證  
- 將 Retention Plan 行動化 (自動通知 / CRM 整合)  
- 建立定期監控流程 (Monthly model refresh, KPI tracking)

---

如需更多協助，歡迎提出 Issue 或直接聯繫專案負責人。  
Feel free to open an issue or ping the project owner for support.



