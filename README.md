# Thyroid Cancer Recurrence Classification

## 📌 Project Overview
This project is part of a **Data Mining** course assignment in the **Master’s Program at Düzce University**. The objective is to replicate the methodology of the following research paper and compare the results with our findings:

> **Machine learning for risk stratification of thyroid cancer patients: a 15-year cohort study**  
> *Shiva Borzooei, Giovanni Briganti, Mitra Golparian, Jerome R. Lechien, Aidin Tarokhian*  
> European Archives of Oto-Rhino-Laryngology (2024) 281:2095–2104  
> [DOI: 10.1007/s00405-023-08299-w](https://doi.org/10.1007/s00405-023-08299-w)  

The goal is to apply the **same classification algorithms** and evaluate the results, discussing potential discrepancies and their causes.

## 🩺 **Dataset**
The dataset used in this study is the **Differentiated Thyroid Cancer Recurrence Dataset**, available at:  
🔗 [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)  

- **Number of instances:** 383 patients  
- **Number of features:** 16  
- **Classification task:** Predict recurrence of thyroid cancer  

## 🛠 **Methods & Preprocessing**
- **Feature Encoding:**
  - **Binary Encoding:** `Gender`, `Smoking`, `Hx Smoking`, `Hx Radiotherapy`, `Recurred`
  - **Ordinal Encoding:** `Risk`, `Focality`, `Stage`, `T`, `N`, `M`
  - **One-Hot Encoding:** `Thyroid Function`, `Physical Examination`, `Pathology`, `Response`, `Adenopathy`
- **Feature Scaling:** StandardScaler applied to all numerical variables
- **Train-Test Split:** 283 training, 100 test instances (following the research paper)
- **Algorithms Used:**
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning:** Grid Search was used to find the optimal hyperparameters for each model before training.

## 📊 Evaluation Metrics
The following metrics were used to evaluate model performance:

| Metric                  | Definition |
|-------------------------|------------|
| **Accuracy**           | (TP + TN) / (TP + TN + FP + FN) |
| **Sensitivity (Recall)** | TP / (TP + FN) |
| **Specificity**         | TN / (TN + FP) |
| **Precision (PPV)**     | TP / (TP + FP) |
| **Negative Predictive Value (NPV)** | TN / (TN + FN) |
| **AUC Score**          | Area under the ROC curve |

## 🔍 **Discussion & Findings**
The results obtained were compared with the original study, and differences were analyzed based on:

- **Training-Test Splitting:** Different patient distributions  
- **Feature Encoding Choices:** Ordinal vs One-Hot Encoding  
- **Hyperparameter Differences:** Selected using **Grid Search**  
- **Standardization Algorithm Variations:** Differences in `StandardScaler` implementation  

## 📂 **Results & Outputs**
After executing the pipeline, the following files will be generated:

📄 `results.xlsx` → Model evaluation metrics  
📄 `hyperparameters.xlsx` → Best hyperparameters found using Grid Search  
📄 `X_train_scaled.xlsx`, `X_test_scaled.xlsx` → Scaled feature datasets  
