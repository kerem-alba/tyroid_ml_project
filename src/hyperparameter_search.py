from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def find_best_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        model, param_grid, cv=2, scoring="accuracy", n_jobs=-1 
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n En İyi Hiperparametreler: {best_params}")
    print(f" En İyi Accuracy: {best_score:.4f}")
    
    return best_params, best_score

# Makaledeki Parametre Aralıkları
rf_param_grid = {
    "n_estimators": [50, 60, 100, 200],
    "max_depth": [3, 4, 5, None],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", None]
}

knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9],  # Makaledeki değerleri aynen koruduk
    "weights": ["uniform", "distance"]
}

svm_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": ["balanced", None]
}

def tune_hyperparameters(X_train, y_train):
    best_rf_params, _ = find_best_hyperparameters(RandomForestClassifier(random_state=42), rf_param_grid, X_train, y_train)
    best_knn_params, _ = find_best_hyperparameters(KNeighborsClassifier(), knn_param_grid, X_train, y_train)
    best_svm_params, _ = find_best_hyperparameters(SVC(random_state=42), svm_param_grid, X_train, y_train)
    return best_rf_params, best_knn_params, best_svm_params

