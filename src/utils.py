import pandas as pd

def save_hyperparameters_to_excel(rf_params, knn_params, svm_params, file_path="hyperparameters.xlsx"):
    """ Hiperparametreleri Excel dosyasına kaydeder. """
    df = pd.DataFrame([
        ["Random Forest", rf_params],
        ["KNN", knn_params],
        ["SVM", svm_params],
    ], columns=["Model", "Best Hyperparameters"])
    
    df.to_excel(file_path, index=False, sheet_name="Best Hyperparameters")
    print(f"📌 Hiperparametreler başarıyla kaydedildi: {file_path}")

def save_results_to_excel(results, file_path="results.xlsx"):
    """ Makale sonuçları ve Grid Search ile hesaplanan sonuçları kaydeder. """
    
    # 📌 Makalede verilen sonuçlar
    makale_results = {
        "Random Forest": (0.9666, 0.9428, 0.8787, 0.9850, 0.9938, 0.95),
        "KNN": (0.8300, 0.9714, 0.9259, 0.9315, 0.9842, 0.93),
        "SVM": (0.9333, 0.9714, 0.9333, 0.9714, 0.9971, 0.96),
    }

    formatted_results = []

    # 📌 Sonuçları sıralı olarak ekleyelim (Makale → Grid Search Sonuçları)
    for model_name, makale_values in makale_results.items():
        # ✅ Makale sonuçlarını ekle
        formatted_results.append([
            model_name, "Makale Sonucu",
            f"{makale_values[0]:.4f}", f"{makale_values[1]:.4f}",
            f"{makale_values[2]:.4f}", f"{makale_values[3]:.4f}",
            f"{makale_values[4]:.4f}", f"{makale_values[5]:.4f}"
        ])

        # ✅ Grid Search ile hesaplanan sonuçları ekle
        for res in results:
            if f"{model_name} (Grid Search)" in res[0]:
                formatted_results.append([
                    model_name, "Grid Search Sonucu",
                    f"{res[1]:.4f}", f"{res[2]:.4f}",
                    f"{res[3]:.4f}", f"{res[4]:.4f}",
                    f"{res[5]:.4f}", f"{res[6]:.4f}"
                ])
    
    # 📌 Excel için DataFrame oluştur
    df_comparison = pd.DataFrame(formatted_results, columns=[
        "Model", "Kaynak", "Sensitivity", "Specificity", "PPV (Precision)", "NPV", "AUC", "Accuracy"
    ])

    # 📌 Excel'e kaydet
    df_comparison.to_excel(file_path, index=False, sheet_name="Sonuç Karşılaştırma")
    print(f"📌 Sonuçlar başarıyla kaydedildi: {file_path}")
