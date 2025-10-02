import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier # YENİ EKLEME
import joblib
import os
import warnings

# --- YOL VE YAPILANDIRMA ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_FILENAME = os.path.join(PROJECT_ROOT, 'models', 'kepler_ai_best_model.joblib') 

def train_and_optimize(X, y, feature_names):
    """XGBoost, LightGBM ve CatBoost modellerini eğitir, karşılaştırır ve en iyisini kaydeder."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )

    models_to_train = {
        'XGBoost': {
            'estimator': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            'param_grid': {'n_estimators': [100, 300], 'max_depth': [10], 'learning_rate': [0.05, 0.1]}
        },
        'LightGBM': {
            'estimator': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
            'param_grid': {'n_estimators': [100, 300], 'max_depth': [10], 'learning_rate': [0.05, 0.1]}
        },
        'CatBoost': { # YENİ CATBOOST TANIMI
            'estimator': CatBoostClassifier(random_state=42, verbose=0, thread_count=-1),
            'param_grid': {
                'iterations': [100, 300],
                'learning_rate': [0.05, 0.1],
                'depth': [10]
            }
        }
    }

    best_overall_model = None
    best_overall_accuracy = 0.0
    best_overall_name = ""
    
    # ... (Karşılaştırma ve Kaydetme Mantığı Aynı Kalır) ...
    # Kodu önceki cevaptan alıp, 'CatBoost'u ekleyerek ve uygun parametreleri ayarlayarak devam ettirin. 

    print("\n--- 3. Grid Search Karşılaştırmalı Optimizasyon Başlatılıyor (XGB, LGBM, CatBoost) ---")

    for name, config in models_to_train.items():
        print(f"\n****** Başlatılıyor: {name} ******")
        
        grid_search = GridSearchCV(
            estimator=config['estimator'], 
            param_grid=config['param_grid'], 
            cv=3, 
            scoring='accuracy', 
            verbose=0, 
            n_jobs=-1             
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # CatBoost için özel eğitim çağrısı (verbose=0'ı yoksayabilir)
            if name == 'CatBoost':
                 # CatBoost için GridSearch çalıştırılır
                 grid_search.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=False)
            else:
                 grid_search.fit(X_train, y_train)

        
        current_best = grid_search.best_estimator_
        y_pred = current_best.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[{name} SONUÇ] En iyi parametreler: {grid_search.best_params_}")
        print(f"[{name} SONUÇ] Doğruluk (Accuracy): {accuracy:.4f}")

        if accuracy > best_overall_accuracy:
            best_overall_accuracy = accuracy
            best_overall_model = current_best
            best_overall_name = name

    # En İyi Modeli Kaydetme
    joblib.dump(best_overall_model, MODEL_FILENAME)
    print(f"\n[BAŞARILI] En İyi Model Kaydedildi ({best_overall_name}, Doğruluk: {best_overall_accuracy:.4f}): {MODEL_FILENAME}")
    
    # En İyi Modelin Nihai Raporu
    y_pred_best = best_overall_model.predict(X_test)
    print("\n--- 4. NİHAİ EN İYİ MODEL PERFORMANSI ---")
    print(f"Model: {best_overall_name}, Doğruluk: {best_overall_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_best))
    
    # Öznitelik Önemi
    try:
        feature_importances = pd.Series(best_overall_model.feature_importances_, index=feature_names)
        print("\n--- 5. Top 10 Öznitelik Önemi ---")
        print(feature_importances.nlargest(10).to_string())
    except AttributeError:
        print(f"\n--- 5. Top 10 Öznitelik Önemi ---\nUyarı: {best_overall_name} modeli öznitelik önemi sağlamıyor.")

    return best_overall_model