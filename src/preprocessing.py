import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert categorical binary variables to numeric
    df["Recurred"] = df["Recurred"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

    # Convert binary columns to numeric
    binary_cols = ["Smoking", "Hx Smoking", "Hx Radiothreapy"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ordinal Encoding
    df["Risk"] = df["Risk"].map({"Low": 0, "Intermediate": 1, "High": 2})
    df["Focality"] = df["Focality"].map({"Uni-Focal": 0, "Multi-Focal": 1})
    
    stage_mapping = {"I": 1, "II": 2, "III": 3, "IVA": 4, "IVB": 5}
    df["Stage"] = df["Stage"].map(stage_mapping)

    t_mapping = {"T1a": 1, "T1b": 2, "T2": 3, "T3a": 4, "T3b": 5, "T4a": 6, "T4b": 7}
    df["T"] = df["T"].map(t_mapping)

    n_mapping = {"N0": 0, "N1a": 1, "N1b": 2}
    df["N"] = df["N"].map(n_mapping)

    m_mapping = {"M0": 0, "M1": 1}
    df["M"] = df["M"].map(m_mapping)

    # One-Hot Encoding (excluding ordinal-encoded columns)
    one_hot_cols = ["Thyroid Function", "Physical Examination", "Pathology", "Response", "Adenopathy"]
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)

    # Split into features and target
    X = df.drop(columns=["Recurred"])
    y = df["Recurred"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=283, test_size=100, random_state=42)

    # ✅ Apply scaling to all features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # ✅ Convert target variables to DataFrame for consistency
    y_train_df = pd.DataFrame(y_train, columns=["Recurred"])
    y_test_df = pd.DataFrame(y_test, columns=["Recurred"])

    # ✅ Save all datasets in one Excel file
    output_file = "C:/Users/ASUS/Desktop/Duzce-CS/BM525-Veri Isleme/Classification/tyroid_ml_project_R/scaled_data.xlsx"

    with pd.ExcelWriter(output_file) as writer:
        X_train_scaled.to_excel(writer, sheet_name="X_train_scaled", index=False)
        X_test_scaled.to_excel(writer, sheet_name="X_test_scaled", index=False)
        y_train_df.to_excel(writer, sheet_name="y_train", index=False)
        y_test_df.to_excel(writer, sheet_name="y_test", index=False)

    print("✅ Scaling completed and saved to Excel.")

    return X_train_scaled, X_test_scaled, y_train, y_test

