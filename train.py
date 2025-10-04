import pandas as pd
import numpy as np
import joblib
import os
import requests
import warnings
import logging
import time
import json
from io import StringIO
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

# --- YAPILANDIRMA VE SABİTLER ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

NASA_KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20*%20from%20cumulative&format=csv"
REQUIRED_COLUMNS = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                    'koi_fpflag_co', 'koi_period', 'koi_depth', 
                    'koi_prad', 'koi_steff', 'koi_disposition'] 
FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col != 'koi_disposition']

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

DISCOVERY_LOG_PATH = os.path.join(MODEL_DIR, "discovery_history.json")

def fetch_data(url: str) -> pd.DataFrame:
    """NASA'dan güncel KOI verisini çeker."""
    logger.info(f"Veri çekiliyor: {url}")
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), comment='#')
        logger.info(f"Veri başarıyla çekildi. Satır sayısı: {len(df)}")
        return df
    except requests.RequestException as e:
        logger.error(f"NASA'dan veri çekme hatası: {e}")
        raise

def preprocess_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Veriyi CatBoost eğitimi için temizler ve hazırlar."""
    df_clean = df.copy()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        logger.error(f"Eksik zorunlu sütunlar: {missing_cols}. Eğitim iptal edildi.")
        raise ValueError(f"Eksik zorunlu sütunlar: {missing_cols}")

    # Hedef değişken: 1 - gezegen adayı/onaylanmış, 0 - yanlış pozitif
    df_clean['is_exoplanet'] = df_clean['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    y = df_clean['is_exoplanet']

    # Sadece özellikler ve hedefi tut
    df_clean = df_clean[FEATURE_COLUMNS + ['is_exoplanet']]

    # Eksik verileri ortalama ile doldur
    df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)

    X = df_clean[FEATURE_COLUMNS]
    logger.info(f"Veri temizleme tamamlandı. Eğitim seti boyutu: {len(X)}")
    return X, y

def detect_new_candidates(df: pd.DataFrame, model, scaler, threshold=0.95) -> List[Dict[str, Any]]:
    """
    Eğitilmiş model ile yüksek güvenli yeni gezegen adaylarını tespit et.
    Sadece CONFIRMED veya CANDIDATE olmayan satırlarda tahmin yapılır.
    """
    candidates = df[df['koi_disposition'].isin(['FALSE POSITIVE']) == False].copy()
    X = candidates[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    pred_proba = model.predict_proba(X_scaled)[:, 1]
    candidates['ai_score'] = pred_proba
    candidates['ai_label'] = np.where(pred_proba > 0.5, "Gezegen/Aday", "Yanlış Pozitif")
    # Sadece yüksek güvenli yeni adaylar
    high_conf = candidates[(candidates['ai_score'] > threshold) & (candidates['ai_label'] == "Gezegen/Aday")]
    logger.info(f"Yüksek güvenle tespit edilen yeni adaylar: {len(high_conf)}")
    # Kısa özet
    discoveries = []
    for _, row in high_conf.iterrows():
        discoveries.append({
            "date": time.strftime("%Y-%m-%d"),
            "kepid": row.get("kepid", ""),
            "koi_id": row.get("koi_id", ""),
            "score": float(row["ai_score"]),
            "label": row["ai_label"],
            "period": row.get("koi_period", ""),
            "radius": row.get("koi_prad", ""),
            "depth": row.get("koi_depth", ""),
            "star_temp": row.get("koi_steff", ""),
            "comment": "Yüksek güvenli yeni gezegen adayı"
        })
    return discoveries

def train_and_save_model(X: pd.DataFrame, y: pd.Series) -> Tuple[CatBoostClassifier, StandardScaler]:
    """Modeli eğitir ve scaler ile birlikte kaydeder."""
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'kepler_ai_scaler.joblib'))
    logger.info("Scaler eğitildi ve kaydedildi.")

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
    joblib.dump(model, os.path.join(MODEL_DIR, 'kepler_ai_best_model.joblib'))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, 'kepler_ai_feature_names.joblib'))

    with open(os.path.join(MODEL_DIR, 'last_trained.txt'), 'w') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(f"Model, scaler ve özellikler '{MODEL_DIR}' klasörüne kaydedildi.")
    return model, scaler

def save_discovery_log(discoveries: List[Dict[str, Any]]):
    """Otomatik keşifleri JSON olarak kaydet."""
    try:
        with open(DISCOVERY_LOG_PATH, 'w') as f:
            json.dump(discoveries, f, indent=2)
        logger.info(f"Keşif geçmişi güncellendi: {DISCOVERY_LOG_PATH}")
    except Exception as e:
        logger.error(f"Keşif geçmişi kaydedilemedi: {e}")

def main():
    """Otonom eğitim ve keşif pipeline."""
    logger.info("--- OTONOM EĞİTİM PIPELINE BAŞLADI ---")
    try:
        raw_df = fetch_data(NASA_KOI_URL)
        X, y = preprocess_and_clean(raw_df)
        model, scaler = train_and_save_model(X, y)
        # Yeni aday tespiti ve loglama
        discoveries = detect_new_candidates(raw_df, model, scaler, threshold=0.95)
        save_discovery_log(discoveries)
        logger.info("--- PIPELINE BAŞARIYLA TAMAMLANDI ---")
    except Exception as e:
        logger.error(f"EĞİTİM SIRASINDA KRİTİK HATA: {e}", exc_info=True)
        logger.error("--- PIPELINE BAŞARISIZ OLDU ---")

if __name__ == "__main__":
    main()