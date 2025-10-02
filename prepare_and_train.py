# prepare_and_train.py

import sys
# Modülleri src klasöründen import et
from src.data_processor import load_data, preprocess_and_scale
from src.model_trainer import train_and_optimize

# --- YAPILANDIRMA ---
# Veri yolu artık data/ klasörünü gösteriyor
DATA_FILE = 'data/cumulative_koi_data.csv' 

if __name__ == "__main__":
    
    try:
        import xgboost # Kurulum kontrolü
    except ImportError:
        print("\n[KRİTİK HATA] XGBoost kütüphanesi kurulu değil. Lütfen 'pip install xgboost' çalıştırın.")
        sys.exit(1)
        
    print("====================================================================")
    print("      KEPLER-AI: PROFESYONEL EĞİTİM BAŞLATILIYOR")
    print("====================================================================")
    
    # 1. Veri İşleme (Mantık artık src/data_processor.py içinde)
    df = load_data(DATA_FILE)
    X_scaled, y, feature_names = preprocess_and_scale(df, is_training=True)
    
    # 2. Model Eğitimi ve Optimizasyonu (Mantık artık src/model_trainer.py içinde)
    train_and_optimize(X_scaled, y, feature_names)

    print("\n====================================================================")
    print("EĞİTİM TAMAMLANDI! Tüm varlıklar models/ klasörüne kaydedildi.")
    print("====================================================================")
