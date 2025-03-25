from src.preprocessing import load_and_preprocess_data
from src.train import train_models
from src.hyperparameter_search import tune_hyperparameters
from src.utils import save_results_to_excel, save_hyperparameters_to_excel

# ✅ 1️⃣ Veriyi Yükle
file_path = "data/Thyroid_Diff.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# ✅ 2️⃣ Grid Search ile En İyi Hiperparametreleri Bul
print("\n🔍 Grid Search ile En İyi Hiperparametreleri Bulunuyor...")
best_rf_params, best_knn_params, best_svm_params = tune_hyperparameters(X_train, y_train)

# 📌 Grid Search Sonuçlarını Excel'e Kaydet
save_hyperparameters_to_excel(best_rf_params, best_knn_params, best_svm_params)

# ✅ 3️⃣ Model Eğitimi ve Test Sonuçları (Sadece Grid Search ile bulunan hiperparametreler)
print("\n🚀 Grid Search ile Bulunan En İyi Parametrelerle Modeller Eğitiliyor...")
all_results_best = train_models(X_train, X_test, y_train, y_test,  
                                best_rf_params, best_knn_params, best_svm_params)

rf_results_best = all_results_best["Random Forest"]
knn_results_best = all_results_best["KNN"]
svm_results_best = all_results_best["SVM"]

# 📌 4️⃣ Sonuçları Excel Dosyasına Kaydet
results = [
    ("Random Forest (Grid Search)", *list(rf_results_best.values())),  
    ("KNN (Grid Search)", *list(knn_results_best.values())),
    ("SVM (Grid Search)", *list(svm_results_best.values())),
]

save_results_to_excel(results)

print("\n✅ Tamamlandı! Sonuçlar ve en iyi hiperparametreler Excel'e kaydedildi.")
