import pandas as pd
import numpy as np
import joblib
import os
import requests
import warnings
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List
import logging
from io import StringIO
import time

# --- YAPILANDIRMA VE SABİTLER ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Geçmişte 404 veren ancak yine de en olası çalışan link olan 'cumulative' tablosunu kullanıyoruz.
# Eğer bu link çalışmazsa, lütfen manuel olarak bulduğunuz linki buraya ekleyelim.
NASA_KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20*%20from%20cumulative&format=csv"

# Modelin kullanacağı zorunlu ve hedef sütunlar
REQUIRED_COLUMNS = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                    'koi_fpflag_co', 'koi_period', 'koi_depth', 
                    'koi_prad', 'koi_steff', 'koi_disposition'] 
FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col != 'koi_disposition']

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# KRİTİK DÜZELTME: Loglama formatı hatası giderildi (levelnames -> levelname)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s') 
logger = logging.getLogger(__name__)

# --- FONKSİYONLAR ---

def fetch_data(url: str) -> pd.DataFrame:
    """NASA'dan güncel KOI verisini çeker."""
    logger.info(f"Veri çekiliyor: {url}")
    try:
        response = requests.get(url, timeout=60) 
        response.raise_for_status() # HTTP hatalarını yakala (404, 400 vb.)
        
        # NASA verileri genellikle CSV olarak çekilir.
        df = pd.read_csv(StringIO(response.text), comment='#')
        logger.info(f"Veri başarıyla çekildi. Başlangıç Satır Sayısı: {len(df)}")
        return df
    except requests.RequestException as e:
        logger.error(f"NASA'dan veri çekme hatası. URL ya hatalı (404) ya da sorgu bozuk (400): {e}")
        raise

def preprocess_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Veriyi CatBoost eğitimi için temizler ve hazırlar."""
    df_clean = df.copy()
    
    # 1. Zorunlu Sütun Kontrolü
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        logger.error(f"Eksik zorunlu sütunlar: {missing_cols}. Eğitim İptal Edildi.")
        raise ValueError(f"Eksik zorunlu sütunlar: {missing_cols}")

    # 2. Hedef Değişkeni Oluşturma (1: Gezegen Adayı/Onaylanmış, 0: Yanlış Pozitif)
    df_clean['is_exoplanet'] = df_clean['koi_disposition'].apply(
        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
    )
    y = df_clean['is_exoplanet']

    # 3. Yalnızca Özellik Sütunlarını ve Hedef Değişkeni Saklama
    df_clean = df_clean[FEATURE_COLUMNS + ['is_exoplanet']]
    
    # 4. Eksik Veri Temizliği (NaN/Boşluk)
    df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)
    
    # 5. Özellik ve Hedef değişkenlerini ayırma
    X = df_clean[FEATURE_COLUMNS]

    logger.info(f"Veri temizleme tamamlandı. Eğitim Seti Boyutu: {len(X)}")
    return X, y

def train_and_save_model(X: pd.DataFrame, y: pd.Series):
    """Modeli eğitir ve scaler ile birlikte kaydeder."""
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Ölçekleyiciyi (Scaler) Eğitme ve Kaydetme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'kepler_ai_scaler.joblib'))
    logger.info("Scaler eğitildi ve kaydedildi.")

    # 2. CatBoost Modelini Eğitme
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        verbose=0,
        random_seed=42
    )
    model.fit(X_train_scaled, y_train)
    logger.info("CatBoost modeli başarıyla eğitildi.")
    
    # 3. Modeli ve Özellik İsimlerini Kaydetme
    joblib.dump(model, os.path.join(MODEL_DIR, 'kepler_ai_best_model.joblib'))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, 'kepler_ai_feature_names.joblib'))
    
    # 4. Eğitim tarihini kaydetme (app.py'de gösterilebilir)
    with open(os.path.join(MODEL_DIR, 'last_trained.txt'), 'w') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        
    logger.info(f"Model, Scaler ve Özellikler '{MODEL_DIR}' klasörüne kaydedildi.")

def main():
    """Ana eğitim akışını yönetir."""
    logger.info("--- OTONOM EĞİTİM PIPELINE BAŞLADI ---")
    
    try:
        # 1. Veri çekme
        raw_df = fetch_data(NASA_KOI_URL)
        
        # 2. Veri temizleme ve hazırlama
        X, y = preprocess_and_clean(raw_df)
        
        # 3. Model eğitimi ve kaydı
        train_and_save_model(X, y)
        
        logger.info("--- EĞİTİM PIPELINE BAŞARIYLA TAMAMLANDI ---")

    except Exception as e:
        logger.error(f"EĞİTİM SIRASINDA KRİTİK HATA: {e}", exc_info=True)
        logger.error("--- EĞİTİM PIPELINE BAŞARISIZ OLDU ---")

if __name__ == "__main__":
    main()