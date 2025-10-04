# src/utils.py

import pandas as pd
import numpy as np
import streamlit as st # Sadece caching dekoratörleri için tutuldu
from typing import Tuple, List

# --- Sabitler (app.py'den kopyalandı) ---
REQUIRED_COLUMNS = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                    'koi_fpflag_co', 'koi_period', 'koi_depth', 
                    'koi_prad', 'koi_steff']

# -------------------------------------------------------------------
# 1. VERİ TEMİZLEME VE VALIDASYON (Eski _validate_and_clean_data gövdesi)
# -------------------------------------------------------------------
@st.cache_data(show_spinner="⚙️ Veri Temizleme ve Validasyon İşleniyor...")
def validate_and_clean_data(df_raw: pd.DataFrame, required_columns: list) -> Tuple[pd.DataFrame, List[str]]:
    """Veriyi temizler, zorunlu sütunları kontrol eder ve hatalı satırları çıkarır."""
    df = df_raw.copy()
    initial_count = len(df)
    issues = []
    
    # Tüm sayısal sütunları zorla sayısal tipe dönüştürme
    for col in required_columns:
        if col in df.columns:
             # Hata Ayıklama: Zorunlu sütunlar için hata yönetimini basitleştirme
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sütun bazlı hataları toplu olarak çıkarma
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned = df.dropna(subset=required_columns)
    dropped_nan_count = initial_count - len(df_cleaned)
    if dropped_nan_count > 0:
        issues.append(f"{dropped_nan_count} satırda temel özelliklerde eksik/geçersiz (NaN/Inf) değer olduğu için çıkarıldı.")
    
    df = df_cleaned.copy()
    
    # Fiziksel Sınır Kontrolleri (Logaritmik dönüşümlerden önce sıfır kontrolü KRİTİK)
    for col, label in [('koi_period', 'yörünge periyodu'), ('koi_prad', 'gezegen yarıçapı'), ('koi_depth', 'geçiş derinliği')]:
        if col in df.columns:
             dropped_count = len(df[df[col] <= 0])
             df = df[df[col] > 0]
             if dropped_count > 0:
                  issues.append(f"{dropped_count} satırda {label} sıfır veya negatif olduğu için çıkarıldı.")

    # Skor Kontrolü
    if 'koi_score' in df.columns:
         dropped_score_count = len(df[(df['koi_score'] < 0) | (df['koi_score'] > 1)])
         df = df[(df['koi_score'] >= 0) & (df['koi_score'] <= 1)]
         if dropped_score_count > 0:
             issues.append(f"{dropped_score_count} satırda skor [0, 1] aralığı dışında olduğu için çıkarıldı.")
        
    # Bayrak Kontrolleri
    fp_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co']
    for col in fp_cols:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int) 
             invalid_flag_count = len(df[(df[col] != 0) & (df[col] != 1)])
             df = df[df[col].isin([0, 1])]
             if invalid_flag_count > 0:
                  issues.append(f"{invalid_flag_count} satırda '{col}' bayrağı 0 veya 1 dışında/geçersiz olduğu için çıkarıldı.")

    final_count = len(df)
    if final_count < initial_count:
        issues.insert(0, f"**Toplam {initial_count - final_count} satır VERİ GÜVENLİK SİSTEMİ tarafından atıldı.**")
    
    return df, issues

# -------------------------------------------------------------------
# 2. ÖZELLİK MÜHENDİSLİĞİ VE HİZALAMA (Eski _feature_engineering_and_alignment gövdesi)
# -------------------------------------------------------------------
@st.cache_data(show_spinner="⚙️ Özellik Mühendisliği ve Hizalama İşleniyor...")
def feature_engineering_and_alignment(df_raw_row: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Yeni özellikler türetir, aykırı değerleri (log) yönetir ve 
    sütunları modelin beklediği sıraya hizalar.
    """
    if df_raw_row.empty:
        return pd.DataFrame(0.0, columns=feature_names, dtype=np.float64)

    df_new = df_raw_row.copy()
    EPSILON = 1e-12 

    # --- Logaritmik Dönüşüm ---
    df_new['R_PRAD_log'] = np.log10(df_new['koi_prad'].replace(0, EPSILON))
    df_new['R_PERIOD_log'] = np.log10(df_new['koi_period'].replace(0, EPSILON))
    df_new['R_DEPTH_log'] = np.log10(df_new['koi_depth'].replace(0, EPSILON))
    
    # --- Türetilmiş Fiziksel Özellikler ---
    df_new['koi_density_proxy'] = df_new['koi_prad'] / (df_new['koi_period'].replace(0, EPSILON) ** (1/3))
    df_new['koi_depth_teff_int'] = df_new['koi_depth'] * df_new['koi_steff']
    
    # --- Hizalama ve Veri Tipi Zorlama ---
    df_aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_names, dtype=np.float64)
    
    for col in df_new.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df_new[col].astype(np.float64) 
            
    return df_aligned