import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import sys
import warnings 

# --- YOL VE YAPILANDIRMA ---
# Betiğin nerede çalıştığından bağımsız olarak proje kök dizinini bulur.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TARGET_COLUMN = 'koi_disposition'      
CSV_SEP = ',' 
CSV_SKIPROWS = 88 

# Varlık yolları
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, 'models', 'kepler_ai_feature_names.joblib')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'kepler_ai_scaler.joblib')

# Modellerin kullanmayacağı sütunlar
COLS_TO_DROP = ['koi_disposition', 'LABEL', 'kepid', 'koi_name', 'kepler_name', 
                'koi_prad_err1', 'koi_prad_err2', 'koi_fmeas_err1', 'koi_fmeas_err2', 'rowid'] 
# Not: koi_srad_err1/2 bu listede olmasa bile hata verdi, çünkü veri setinizde bu isimler yok.

def load_data(data_path):
    """Veri setini esnek bir şekilde yükler."""
    
    full_data_path = os.path.join(PROJECT_ROOT, data_path) 
    print(f"  > Veri Yükleniyor: {full_data_path}")
    
    try:
        df = pd.read_csv(full_data_path, sep=CSV_SEP, skiprows=CSV_SKIPROWS, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"[KRİTİK HATA] Veri dosyası bulunamadı: {full_data_path}. Lütfen data/ klasörünü kontrol edin.")
        sys.exit(1)
    except Exception as e:
        print(f"[KRİTİK HATA] Veri yüklenirken hata: {e}")
        sys.exit(1)


def preprocess_and_scale(df, is_training=True):
    """Veriyi temizler, yeni öznitelikler oluşturur, etiketler ve ölçekler."""
    
    # 1. Etiketleme ve Temizleme
    df_filtered = df.dropna(subset=[TARGET_COLUMN]).copy()
    
    # --- YENİ ÖZNİTELİK MÜHENDİSLİĞİ (GÜVENLİ SÜRÜM) ---
    
    EPSILON = 1e-6
    
    # 1. Gezegen Yoğunluğu Tahmini
    df_filtered['koi_density_proxy'] = df_filtered['koi_prad'] / (df_filtered['koi_period'].replace(0, EPSILON) ** (1/3))
    
    # 2. Önemli Özelliklerin Etkileşimi (koi_steff düzeltildi!)
    # Yıldızın etkin sıcaklığı (koi_steff) ile derinliğin çarpımı
    df_filtered['koi_depth_teff_int'] = df_filtered['koi_depth'] * df_filtered['koi_steff'] 
    
    # HATA VEREN koi_srad_err_range ÖZELLİĞİ BURADAN KALDIRILDI!
    
    # --- ÖZNİTELİK MÜHENDİSLİĞİ BİTİŞ ---
    
    # Hedef Sütunu Etiketleme
    positive_labels = ['CONFIRMED', 'CANDIDATE']
    df_filtered['LABEL'] = df_filtered[TARGET_COLUMN].apply(lambda x: 1 if x in positive_labels else 0)

    # Özniteliklerin Seçimi
    # Yeni oluşturulanlar da dahil tüm sayısal sütunları yakalar
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in COLS_TO_DROP]
    
    # Veri setini ayırma ve NaN doldurma
    X = df_filtered[feature_cols].fillna(df_filtered[feature_cols].mean())
    y = df_filtered['LABEL']

    # 2. Ölçekleme ve Kayıt
    if is_training:
        scaler = StandardScaler()
        
        # RuntimeWarning'ları yakalamak ve bastırmak için
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_scaled = scaler.fit_transform(X)
        
        # Varlıkları güvenilir yola kaydet
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_cols, FEATURE_NAMES_PATH)
        print(f"  > Veri Ölçekleyici ve {len(feature_cols)} Öznitelik Adı Kaydedildi.")
    else:
        # Tahmin için önceden eğitilmiş scaler'ı yükle
        try:
            scaler = joblib.load(SCALER_PATH)
            X_scaled = scaler.transform(X)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tahmin için ölçekleyici dosyası bulunamadı: {SCALER_PATH}. Lütfen önce prepare_and_train.py ile modeli eğitin.")
        
    return X_scaled, y, feature_cols