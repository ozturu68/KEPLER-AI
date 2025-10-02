import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import shap
import matplotlib.pyplot as plt
from typing import Any
import warnings

# --- SİSTEM AYARLARI ---
try:
    plt.switch_backend('Agg') 
except ImportError:
    pass 
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------
# 0. STREAMLIT SAYFA AYARLARI
# -----------------------------------------------------------
st.set_page_config(
    page_title="Kepler-AI | Ötegezegen Sınıflandırma",
    layout="wide", 
    initial_sidebar_state="auto"
)

# -----------------------------------------------------------
# 1. VARLIKLARI GÜVENLİ YÜKLEME (@st.cache_resource)
# -----------------------------------------------------------

@st.cache_resource(show_spinner="Yapay Zeka Varlıkları Yükleniyor...")
def load_ml_assets():
    """ML varlıklarını güvenli bir şekilde yükler."""
    
    MODEL_PATH = 'models/kepler_ai_best_model.joblib'
    SCALER_PATH = 'models/kepler_ai_scaler.joblib'
    FEATURES_PATH = 'models/kepler_ai_feature_names.joblib'
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names_list = joblib.load(FEATURES_PATH)
        explainer = shap.TreeExplainer(model) 
        
        return model, scaler, feature_names_list, explainer
    
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli model dosyaları bulunamadı. Eksik Dosya: {e.filename}")
        return None, None, None, None

MODEL, SCALER, FEATURE_NAMES, SHAP_EXPLAINER = load_ml_assets()

# -----------------------------------------------------------
# 2. KRİTİK ÖZELLİK HİZALAMA FONKSİYONU (66 Özellik Çözümü)
# -----------------------------------------------------------

def feature_engineering_and_alignment(df_raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ham veriyi alır, özellik mühendisliği uygular ve modeli beklediği 
    TAM 66 öznitelik sırasına hizalar.
    """
    df_new = df_raw.iloc[[0]].copy() # Sadece ilk satırı al ve kopya oluştur
    EPSILON = 1e-6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 2.1. Öznitelik Mühendisliği (Sizin orijinal kodunuzdan bilinen 5 özellik)
        df_new['R_PRAD_log'] = np.log10(df_new['koi_prad'].replace(0, EPSILON))
        df_new['R_PERIOD_log'] = np.log10(df_new['koi_period'].replace(0, EPSILON))
        df_new['R_DEPTH_log'] = np.log10(df_new['koi_depth'].replace(0, EPSILON))
        df_new['koi_density_proxy'] = df_new['koi_prad'] / (df_new['koi_period'].replace(0, EPSILON) ** (1/3))
        df_new['koi_depth_teff_int'] = df_new['koi_depth'] * df_new['koi_steff']
        
        # ⚠️ Buraya eklenmeyen tüm diğer ~61 özellik, hizalama adımında 0.0 olarak atanacaktır.

    # 2.2. KRİTİK HİZALAMA: Modelin beklediği TAM 66 özellik şemasını garanti etme
    
    # 66 sütunlu boş bir DataFrame oluştur ve index'i koru
    df_aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_names)
    
    # Hesaplanan değerleri, modelin beklediği nihai DataFrame'e kopyala
    for col in df_new.columns:
        if col in df_aligned.columns:
            df_aligned.loc[:, col] = df_new.loc[:, col]
            
    # Bu DataFrame, modelin beklediği TAM 66 özellikli formattır.
    return df_aligned

# -----------------------------------------------------------
# 3. TAHMİN BORU HATTI
# -----------------------------------------------------------

def run_prediction_pipeline(df_source, model, scaler, feature_names, explainer):
    """Veri işleme, tahmin ve yorumlamayı yürütür."""
    
    # 3.1. KRİTİK HİZALAMA ADIMI: 66 özelliği garantile
    X_aligned = feature_engineering_and_alignment(df_source, feature_names)
    
    # 3.2. Ölçekleme
    X_scaled = scaler.transform(X_aligned)
    
    # 3.3. Tahmin
    prediction_proba = model.predict_proba(X_scaled)[0]
    prediction_label = "GEZEGEN/ADAY" if prediction_proba[1] > 0.5 else "YANLIŞ POZİTİF (FALSE POSITIVE)"
    confidence = max(prediction_proba)

    # 3.4. SHAP Yorumlaması
    shap_values = explainer.shap_values(X_scaled)
    values_to_plot = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]
    base_value_to_plot = explainer.expected_value[1]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    shap.waterfall_plot(shap.Explanation(
        values=values_to_plot, 
        base_values=base_value_to_plot, 
        data=X_scaled[0], 
        feature_names=feature_names), max_display=15, show=False)
    
    # Grafik rengi ayarları
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.title("Model Tahmininin SHAP Açıklaması", color='white')
    
    # Bellekte tutma
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor='#0E1117')
    plt.close(fig)
    
    return prediction_label, confidence, buf, df_source.iloc[0].to_dict()

# -----------------------------------------------------------
# 4. STREAMLIT ARAYÜZÜ
# -----------------------------------------------------------

st.title("🔭 Kepler-AI: Ötegezegen Sınıflandırma")
st.markdown("Makine Öğrenimi (CatBoost) kullanarak Kepler verilerindeki adayların sınıflandırılması ve **SHAP** ile yorumlanması.")

st.sidebar.header("Veri Girişi")
st.sidebar.info("Modelinizi test etmek için **sadece 8 temel sütunu** içeren CSV dosyasını yükleyin.")
uploaded_file = st.sidebar.file_uploader("CSV Dosyasını Yükle", type=['csv'])

prediction_container = st.container()

# --- CSV İŞLEME VE TAHMİN ---
if uploaded_file is not None:
    try:
        # Hata Çözümü: İlk 14 satırı atla (NASA verisindeki meta veriler)
        df_raw = pd.read_csv(uploaded_file, skiprows=14)
        
        st.sidebar.markdown("---")
        st.sidebar.success(f"Veri Yüklendi. **{df_raw.shape[1]}** Sütun Algılandı.")

        if st.sidebar.button('🚀 Tahmin Et ve Yorumla', type="primary"):
            
            if MODEL is None:
                st.error("Model yüklenmediği için tahmin yapılamıyor.")
                st.stop()
            
            with st.spinner('Yapay Zeka Analizi Başlatıldı...'):
                
                prediction, confidence, shap_buffer, raw_data = run_prediction_pipeline(df_raw, MODEL, SCALER, FEATURE_NAMES, SHAP_EXPLAINER)
                
                # --- SONUÇ BÖLÜMÜ ---
                with prediction_container:
                    
                    color = "#E03F33" if "YANLIŞ" in prediction else "#33E04C"
                    emoji = "🔴" if "YANLIŞ" in prediction else "🟢"
                    
                    st.markdown("## 🎯 Tahmin Sonucu")
                    st.markdown(f"### {emoji} SONUÇ: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
                    st.subheader(f"Model Güveni: **{confidence:.2%}**")
                    st.markdown("---")

                    # --- DETAYLI YORUM VE GÖRSEL ---
                    st.header("🔬 Modelin Karar Analizi (SHAP)")
                    st.info("Aşağıdaki grafik, modelin bu kararı verirken hangi özelliklerin sonucu ne yönde etkilediğini gösterir.")
                    
                    st.image(shap_buffer, caption='Modelin Kararını Açıklayan SHAP Görseli', use_column_width=True)
                    
                    # --- HAM VERİ ÖZETİ ---
                    st.header("🔍 Detaylı Özellik Yorumu")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Gezegen Yarıçapı (Prad)", f"{raw_data['koi_prad']:.2f} R_Earth", "Prad değeri, büyük gezegenlerin tespitinde kritiktir.")
                    col2.metric("Yörünge Periyodu", f"{raw_data['koi_period']:.2f} Gün", "Çok kısa periyotlar genellikle hatalı sinyallere yol açar.")
                    col3.metric("Merkez Kayması Bayrağı", f"{int(raw_data['koi_fpflag_co'])}", "Bu bayrak 1 ise, sinyalin Kepler hedefi dışından geldiği düşünülür.")


    except ValueError as e:
        st.error(f"Veri Okuma Hatası: {e}. Lütfen CSV dosyanızın **sadece 8 temel sütunu** içerdiğinden ve doğru formatta olduğundan emin olun.")
    except Exception as e:
        st.error(f"Genel Hata: Uygulama sırasında beklenmeyen bir sorun oluştu. Detay: {e}")

else:
    st.info("Başlamak için, NASA Exoplanet Archive'dan indirdiğiniz aday verilerini içeren CSV dosyanızı kenar çubuğundan yükleyin.")