# config.py - Hiperparametreler için merkezi dosya

# 📌 Makaledeki En İyi Hiperparametreler
ARTICLE_RF_PARAMS = {
    "class_weight": "balanced",
    "criterion": "gini",
    "max_depth": 4,
    "max_features": "sqrt",
    "n_estimators": 60
}

ARTICLE_KNN_PARAMS = {
    "n_neighbors": 6,
    "weights": "distance"
}

ARTICLE_SVM_PARAMS = {
    "C": 1,
    "class_weight": None,
    "gamma": "scale",
    "kernel": "rbf"
}

# 📌 Bizim Grid Search ile Bulduğumuz En İyi Hiperparametreler
BEST_RF_PARAMS = {
    "criterion": "gini",
    "max_depth": 3,
    "max_features": None,
    "n_estimators": 60
}

BEST_KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance"
}

BEST_SVM_PARAMS = {
    "C": 0.1,
    "gamma": "scale",
    "kernel": "linear"
}
