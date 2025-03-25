from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

def calculate_metrics(y_true, y_pred, y_proba):
    """ Sensitivity, Specificity, PPV, NPV, AUC, Accuracy hesaplar. """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = None  # Tek sınıf varsa AUC hesaplanamaz

    metrics = {
        "Sensitivity (Recall)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "PPV (Precision)": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred)
    }

    return {k: round(v, 4) if v is not None else None for k, v in metrics.items()}

def train_models(X_train, X_test, y_train, y_test, rf_params, knn_params, svm_params):
    rf_model = RandomForestClassifier(**rf_params)
    knn_model = KNeighborsClassifier(**knn_params)
    svm_model = SVC(**svm_params, probability=True)  # AUC için probability=True gerekli!

    print("\n📌 Training and evaluating models...")

    models = {
        "Random Forest": rf_model,
        "KNN": knn_model,
        "SVM": svm_model
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Min-max scaling
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        
        print(f"\n📊 {name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    
    return results
