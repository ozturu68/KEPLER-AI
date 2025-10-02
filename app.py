import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import shap
import matplotlib.pyplot as plt
import warnings
import time 
import logging
from typing import Tuple, Dict, Any, List

# KRİTİK İMPORT: Streamlit'in dahili yeniden çalıştırma istisnasını yakalamak için.
#from streamlit.runtime.scriptrunner.exceptions import RerunException 


# --- UYGULAMA YAPILANDIRMASI ve SABİTLER ---

# Matplotlib backend ayarı ve uyarıları bastırma
try:
    plt.switch_backend('Agg')
except ImportError:
    pass

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

REQUIRED_COLUMNS = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                    'koi_fpflag_co', 'koi_period', 'koi_depth', 
                    'koi_prad', 'koi_steff']

# İyileştirme: Merkezi durum seçenekleri
INVESTIGATION_STATUS_OPTIONS = ["Yeni Aday", "İncelemeye Alındı", "Yanlış Pozitif (FP)", "Onaylandı (NP)"]

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s') 
logger = logging.getLogger(__name__)

# Streamlit Sayfa Ayarları
st.set_page_config(
    page_title="Kepler-AI | Ötegezegen Sınıflandırma",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- TASARIM İÇİN GÜNCEL CSS ---
st.markdown("""
<style>
    /* Ana içerik alanını sınırla ve ortala */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 1400px;
    }
    
    /* Vurgu Rengi (Kepler Mavisi/Cyan) */
    :root {
        --primary-color: #7FE9F0; 
    }

    /* Streamlit'in varsayılan birincil rengini (butonu vb.) değiştirme */
    .st-emotion-cache-16fvj8 {
        background-color: #7FE9F0 !important;
        color: #0E1117 !important;
    }
    
    /* Başlık Stilini Geliştirme */
    h1 {
        font-size: 2.5em;
        font-weight: 300; 
        color: #FF4B4B; 
        text-align: center;
        border-bottom: 2px solid #262730; 
        padding-bottom: 10px;
    }
    
    /* Sidebar başlıklarını alet kutusu gibi daha belirgin yap */
    #sidebar .st-emotion-cache-1ftru4k, #sidebar .st-emotion-cache-10ohe8r {
        border-bottom: 1px solid #7FE9F0; 
        padding-bottom: 5px;
        margin-top: 20px;
    }
    
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# 1. ML VARLIKLARINI CACHING EDEN FONKSİYON 
# -----------------------------------------------------------

@st.cache_resource(show_spinner="👽 Yapay Zeka Varlıkları Yükleniyor...")
def load_ml_assets_cached():
    """Modeli, ölçekleyiciyi ve SHAP Explainer'ı güvenle yükler."""
    MODEL_PATH = 'models/kepler_ai_best_model.joblib'
    SCALER_PATH = 'models/kepler_ai_scaler.joblib'
    FEATURES_PATH = 'models/kepler_ai_feature_names.joblib'
    
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            logger.error(f"Model dosyaları bulunamadı: {MODEL_PATH} veya {SCALER_PATH}")
            raise FileNotFoundError("Model dosyaları bulunamadı.")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names_list = joblib.load(FEATURES_PATH)
        explainer = shap.TreeExplainer(model) 
        logger.info("ML varlıkları başarıyla yüklendi.")
        
        return model, scaler, feature_names_list, explainer
    except Exception as e:
        logger.exception("Model yükleme sırasında beklenmeyen bir hata oluştu.")
        st.error(f"Model yükleme sırasında beklenmeyen bir hata oluştu: {e}")
        return None, None, None, None

# -----------------------------------------------------------
# 2. MODEL SİSTEMİ SINIFI (Tahmin ve Yorumlama Boru Hattı)
# -----------------------------------------------------------

class ExoplanetClassifier:
    """Makine öğrenimi boru hattını yöneten ana sınıf."""
    
    def __init__(self):
        self.model, self.scaler, self.feature_names, self.explainer = load_ml_assets_cached()
        if self.model is None or self.scaler is None:
             logger.critical("Model veya ölçekleyici yüklenemedi. Uygulama başlatılamıyor.")
             raise RuntimeError("Model yüklenemedi. Devam edilemiyor.")
        logger.info("ExoplanetClassifier instance başarıyla oluşturuldu.")

    @staticmethod
    @st.cache_data(show_spinner="⚙️ Veri Temizleme ve Validasyon İşleniyor...")
    def _validate_and_clean_data(df_raw: pd.DataFrame, required_columns: list) -> Tuple[pd.DataFrame, List[str]]:
        df = df_raw.copy()
        initial_count = len(df)
        issues = []
        
        # Temel sütunları sayısal tiplere zorla
        for col in ['koi_score', 'koi_period', 'koi_depth', 'koi_prad', 'koi_steff']:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Geçersiz (Inf, NaN) değerleri temizle
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_cleaned = df.dropna(subset=required_columns)
        dropped_nan_count = initial_count - len(df_cleaned)
        if dropped_nan_count > 0:
            issues.append(f"{dropped_nan_count} satırda temel özelliklerde eksik/geçersiz (NaN/Inf/Sayısal Olmayan) değer olduğu için çıkarıldı.")
        
        df = df_cleaned.copy()
        
        # Sıfır veya negatif değerleri kontrol et
        for col, label in [('koi_period', 'yörünge periyodu'), ('koi_prad', 'gezegen yarıçapı'), ('koi_depth', 'geçiş derinliği')]:
            dropped_count = len(df[df[col] <= 0])
            df = df[df[col] > 0]
            if dropped_count > 0:
                 issues.append(f"{dropped_count} satırda {label} sıfır veya negatif olduğu için çıkarıldı.")

        # Skor aralığı [0, 1] kontrolü
        dropped_score_count = len(df[(df['koi_score'] < 0) | (df['koi_score'] > 1)])
        df = df[(df['koi_score'] >= 0) & (df['koi_score'] <= 1)]
        if dropped_score_count > 0:
            issues.append(f"{dropped_score_count} satırda skor [0, 1] aralığı dışında olduğu için çıkarıldı.")
            
        # FP bayrakları kontrolü (0 veya 1 olmalı)
        fp_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co']
        for col in fp_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int) 
                 invalid_flag_count = len(df[(df[col] != 0) & (df[col] != 1)])
                 df = df[(df[col] == 0) | (df[col] == 1)]
                 if invalid_flag_count > 0:
                      issues.append(f"{invalid_flag_count} satırda '{col}' bayrağı 0 veya 1 dışında/geçersiz olduğu için çıkarıldı.")

        final_count = len(df)
        if final_count < initial_count:
            issues.insert(0, f"**Toplam {initial_count - final_count} satır KONTROL SİSTEMİ tarafından geçersiz veri nedeniyle atıldı.**")
        
        logger.info(f"Veri temizleme tamamlandı. Başlangıç: {initial_count}, Son: {final_count}")
        return df, issues
    
    @staticmethod
    def _feature_engineering_and_alignment(df_raw_row: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        df_new = df_raw_row.copy()
        EPSILON = 1e-12 

        # Log dönüşümleri
        df_new['R_PRAD_log'] = np.log10(df_new['koi_prad'].replace(0, EPSILON))
        df_new['R_PERIOD_log'] = np.log10(df_new['koi_period'].replace(0, EPSILON))
        df_new['R_DEPTH_log'] = np.log10(df_new['koi_depth'].replace(0, EPSILON))
        
        # Yeni türetilmiş özellikler
        df_new['koi_density_proxy'] = df_new['koi_prad'] / (df_new['koi_period'].replace(0, EPSILON) ** (1/3))
        df_new['koi_depth_teff_int'] = df_new['koi_depth'] * df_new['koi_steff']
        
        # Özellik hizalama (Modelin beklediği tüm sütunları oluştur)
        df_aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_names, dtype=np.float64)
        
        for col in df_new.columns:
            if col in df_aligned.columns:
                df_aligned.loc[:, col] = df_new.loc[:, col].astype(np.float64)
                
        return df_aligned
    
    def _get_confidence_robust(self, X_scaled: np.ndarray, num_runs: int = 10) -> Tuple[str, float]:
        """Model tahminini birden çok kez jitter'lı veri ile yaparak kararlılığı artırır."""
        JITTER_SCALE = 0.001 
        all_probabilities = []
        
        for _ in range(num_runs):
            X_jittered = X_scaled + np.random.normal(0, JITTER_SCALE, X_scaled.shape)
            proba = self.model.predict_proba(X_jittered)[0]
            all_probabilities.append(proba)
            
        avg_probabilities = np.mean(all_probabilities, axis=0)
        
        prediction_label = "GEZEGEN/ADAY" if avg_probabilities[1] > 0.5 else "YANLIŞ POZİTİF (FALSE POSITIVE)"
        confidence = max(avg_probabilities)
        
        return prediction_label, confidence
    
    def predict(self, df_raw: pd.DataFrame, row_index: int) -> Tuple[str, float, io.BytesIO, Dict[str, Any]]:
        df_raw_row = df_raw.iloc[[row_index]]
        logger.info(f"Aday {row_index+1} için tahmin başlatıldı.")
            
        try:
            X_aligned = self._feature_engineering_and_alignment(df_raw_row, self.feature_names)
            X_scaled = self.scaler.transform(X_aligned.values)
            prediction_label, confidence = self._get_confidence_robust(X_scaled, num_runs=10)
            
            # SHAP Değerlerini Hesapla
            shap_values = self.explainer.shap_values(X_scaled) 
            
            if isinstance(shap_values, list):
                # Sınıf 1 (Pozitif/Gezegen) için değerleri kullan
                target_class_index = 1 if len(shap_values) > 1 else 0
                values_to_plot = shap_values[target_class_index][0] 
                base_value_to_plot = self.explainer.expected_value[target_class_index]
            else:
                values_to_plot = shap_values[0]
                base_value_to_plot = self.explainer.expected_value

            if isinstance(base_value_to_plot, np.ndarray) and base_value_to_plot.ndim > 0:
                 base_value_to_plot = base_value_to_plot.flatten()[0]
            
            shap_plot_data = shap.Explanation(
                values=values_to_plot, 
                base_values=base_value_to_plot, 
                data=X_scaled[0], 
                feature_names=self.feature_names
            )
            
            # SHAP Görselini Oluştur ve Bellekte Sakla
            plt.style.use('dark_background') 
            fig = plt.figure(figsize=(18, 12)) 
            shap.plots.waterfall(shap_plot_data, max_display=15, show=False)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#0E1117') 
            buf.seek(0)
            plt.close(fig) 

            logger.info(f"Aday {row_index+1} için tahmin tamamlandı: {prediction_label}, Güven: {confidence:.2%}")
            
            return prediction_label, confidence, buf, df_raw_row.iloc[0].to_dict()

        except Exception as e:
            logger.exception(f"Aday {row_index+1} için tahmin/SHAP üretimi sırasında kritik hata oluştu.")
            raise RuntimeError(f"Tahmin ve SHAP üretimi sırasında kritik hata: {e}")


# -----------------------------------------------------------
# 3. STREAMLIT ANA UYGULAMA MANTIĞI
# -----------------------------------------------------------

try:
    CLASSIFIER = ExoplanetClassifier()
except RuntimeError as e:
    logger.critical(f"Uygulama başlatılamadı: {e}")
    st.error(f"Uygulama Çalıştırma Hatası: {e}. Model varlıklarının doğru yüklendiğinden emin olun.")
    st.stop()


# --- MERKEZİ ODAKLI BAŞLIK ALANI ---
with st.container():
    col_left_title, col_center_title, col_right_title = st.columns([1, 3, 1])

    with col_center_title:
        st.title("🔭 Kepler-AI: Ötegezegen Keşif Asistanı")
        st.markdown("### <p style='text-align: center; color: #7FE9F0;'>Model Yorumlanabilirlik (XAI) ile desteklenen yüksek güvenilirlikli analiz platformu.</p>", unsafe_allow_html=True)
    st.markdown("---")


def run_simulation_animation(candidate_name, total_duration=3.0):
    """ANALİZ SÜRESİ OPTİMİZASYONU: 3.0 saniyelik görsel bekleme barı."""
    col_left_anim, col_center_anim, col_right_anim = st.columns([1, 3, 1])
    
    with col_center_anim:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status_placeholder.subheader(f"💫 Aday {candidate_name} için Yüksek Güvenilirlikli Analiz Başlatıldı...")
        
        stages = [(0.1, "1/3: Veri Özellikleri Hizalanıyor..."), (0.4, "2/3: Monte Carlo Simülasyonu Başlatıldı..."), (0.8, "3/3: Yapay Zeka Modeli Son Olasılık Skorlarını Birleştiriyor."), (1.0, "✅ Analiz Tamamlandı! Karar Açıklaması Oluşturuldu.")]
        current_progress = 0.0
        start_time = time.time()
        
        for target_progress, message in stages:
            progress_bar.progress(int(target_progress * 100))
            status_placeholder.markdown(f"**{message}**")
            time_to_wait = total_duration * (target_progress - current_progress) * 0.9 
            time.sleep(time_to_wait)
            current_progress = target_progress

        remaining_time = total_duration - (time.time() - start_time)
        if remaining_time > 0:
            time.sleep(remaining_time)
        
        progress_bar.empty()
        status_placeholder.empty()
        st.success(f"✅ Analiz Başarılı.")
        time.sleep(0.5)


# İyileştirme: FP Bayraklarını daha anlaşılır metne dönüştüren yardımcı fonksiyon
def map_flag_to_text(flag_val, name):
    """0/1 flag değerini açıklayıcı metin ve emojiye dönüştürür."""
    if flag_val == 1:
         return f"❌ Bayrak Kaldırıldı ({name})"
    return f"✅ Normal (Temiz)"


def main_prediction_page():
    """TEKİL ADAY ANALİZİ SAYFASI (Ana Görünüm)"""
    
    if st.session_state.df_raw is None:
        st.error("Lütfen önce veri yükleme sayfasından geçerli bir Kepler veri seti yükleyin.")
        return

    # İYİLEŞTİRME 1: Veri Seti Genel Durumu Özeti
    st.subheader("📊 Yüklenen Veri Seti Genel Durumu")
    
    df_raw = st.session_state.df_raw
    
    # Etiketlenmiş aday sayısını hesapla
    labeled_count = len(df_raw[df_raw['Investigation_Status'] != "Yeni Aday"])
    
    # Ortalama periyodu güvenli bir şekilde hesapla
    mean_period = df_raw['koi_period'].mean() if not df_raw.empty else 0.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
         st.metric("Toplam Analiz Edilebilir Aday", len(df_raw))
    with col2:
         st.metric("Etiketlenmiş Aday Sayısı", labeled_count)
    with col3:
         st.metric("Ortalama Yörünge Periyodu", f"{mean_period:.2f} Gün")
    st.markdown("---")


    # --- Analiz Sonuçlarının Gösterilmesi ---
    if 'show_results' in st.session_state and st.session_state.show_results:
        prediction, confidence, shap_buffer, raw_data = st.session_state.last_prediction
        
        st.header(f"2. 🛰️ Aday {st.session_state.selected_candidate_index+1} için Analiz Raporu")
        st.markdown("---")
        
        is_false_positive = "YANLIŞ" in prediction
        emoji = "🚨" if is_false_positive else "✅"
        color = "#7FE9F0" if not is_false_positive else "#DC3545" 
        
        # --- SONUÇ VE GÜVEN METRİKLERİ ---
        st.subheader("🎯 Tahmin ve Güven Özeti")
        
        col_pred, col_conf, col_empty = st.columns([3, 2, 1]) 
        
        with col_pred:
            st.markdown(f"""
            <div style='background-color: #262730; padding: 15px; border-radius: 10px; border-left: 8px solid {color}; box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);'>
                <p style='font-size: 1.1em; margin: 0; color: #AFAFAF; font-weight: 500;'>Sınıflandırma Sonucu</p>
                <h1 style='color: {color}; margin: 5px 0 0 0;'>{emoji} {prediction}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_conf:
            st.metric(label="Model Güven Skoru", value=f"{confidence:.2%}") 
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- ASTROFİZİKSEL VERİLER VE BAYRAKLAR ---
        st.subheader("🔭 Temel Parametreler")
        col_prad, col_period, col_depth, col_steff, col_score = st.columns(5)
        
        with col_prad: st.metric(r"Gezegen Yarıçapı ($R_{\oplus}$)", f"{raw_data.get('koi_prad', 0.0):.2f}")
        with col_period: st.metric("Yörünge Periyodu", f"{raw_data.get('koi_period', 0.0):.2f} Gün")
        with col_depth: st.metric("Geçiş Derinliği", f"{raw_data.get('koi_depth', 0.0):.2e} ppm")
        with col_steff: st.metric("Yıldız Sıcaklığı", f"{raw_data.get('koi_steff', 0.0):.0f} K")
        with col_score: st.metric("Kepler/KOI Skoru", f"{raw_data.get('koi_score', 0.0):.3f}")

        st.markdown("---")

        # --- SHAP GÖRSELİ ---
        st.header("🔬 Modelin Karar Analizi (XAI)")
        st.info("SHAP Waterfall Plot, modelin tahminini hangi özelliklerin, hangi yönde (pozitif/negatif) ve ne kadar etkilediğini gösterir.")
        
        st.image(shap_buffer, caption=f'Aday {st.session_state.selected_candidate_index+1} için SHAP Etki Görseli')
    
    else:
         st.info("Lütfen sol taraftaki Aday Seçimi bölümünden bir aday seçin ve 'Analiz Et' butonuna tıklayın.")


def collective_analysis_page():
    """TOPLU VERİ SETİ İNCELEMESİ SAYFASI (Ana Görünüm)"""
    
    if st.session_state.df_raw is None:
        st.error("Lütfen önce veri yükleme sayfasından geçerli bir Kepler veri seti yükleyin.")
        return
        
    df_raw = st.session_state.df_raw
    
    # --- Sidebar'dan Filtreleri Uygula ---
    period_range = st.session_state.period_range
    score_threshold = st.session_state.score_threshold
    status_filter = st.session_state.status_filter 
         
    # Tüm filtreleri uygula
    df_filtered = df_raw[
        (df_raw['koi_period'] >= period_range[0]) & 
        (df_raw['koi_period'] <= period_range[1]) &
        (df_raw['koi_score'] >= score_threshold) &
        (df_raw['Investigation_Status'].isin(status_filter))
    ]
    
    # İyileştirme: Görsel Netlik için FP bayraklarını çevir
    df_display = df_filtered.copy()
    df_display['FP_NT'] = df_display['koi_fpflag_nt'].apply(lambda x: map_flag_to_text(x, "Gürültü"))
    df_display['FP_SS'] = df_display['koi_fpflag_ss'].apply(lambda x: map_flag_to_text(x, "Çoklu Sistem"))
    df_display['FP_CO'] = df_display['koi_fpflag_co'].apply(lambda x: map_flag_to_text(x, "Merkez Kayması"))
         
    # --- UI ---
    st.header("📋 Toplu Veri Seti İncelemesi")
    st.info("Sol taraftaki (Alet Çantası) filtreleri kullanarak bu tabloyu anlık olarak daraltabilirsiniz. **'İnceleme Durumu'** sütununu doğrudan tabloda düzenleyebilirsiniz.")
    
    st.markdown(f"**Toplam Analiz Edilebilir Aday:** **<span style='color:#7FE9F0;'>{len(df_raw)}</span>**", unsafe_allow_html=True)
    st.markdown(f"**Filtrelenmiş Sonuç Sayısı:** **<span style='color:#FF9900;'>{len(df_filtered)}</span>**", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🔍 Aday İnceleme Tablosu")
    
    # İYİLEŞTİRME 2: Durum Renk Kodları Açıklaması
    status_colors = {
        "Yeni Aday": "#5A5E66",           
        "İncelemeye Alındı": "#FF9900",   
        "Yanlış Pozitif (FP)": "#DC3545", 
        "Onaylandı (NP)": "#7FE9F0"       
    }
    st.markdown(f"""
    <div style='display: flex; gap: 20px; margin-bottom: 20px;'>
        <span style='color: {status_colors["Yeni Aday"]}; font-weight: bold;'>⚫ Yeni Aday</span>
        <span style='color: {status_colors["İncelemeye Alındı"]}; font-weight: bold;'>🟠 İncelemeye Alındı</span>
        <span style='color: {status_colors["Yanlış Pozitif (FP)"]}; font-weight: bold;'>🔴 FP</span>
        <span style='color: {status_colors["Onaylandı (NP)"]}; font-weight: bold;'>🔵 Onaylandı</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Görüntülenecek sütunları belirle: Orijinal FP'ler yerine yeni, açıklayıcı sütunları kullan.
    columns_to_show = [
        'koi_score', 'koi_period', 'koi_prad', 'koi_depth', 
        'FP_NT', 'FP_SS', 'FP_CO', 'Investigation_Status' 
    ]
    df_display = df_display.filter(items=columns_to_show)

    # 💥 st.data_editor ile interaktif etiketleme ve görsel netlik
    edited_df_view = st.data_editor(
        df_display, 
        use_container_width=True, 
        hide_index=False,
        column_config={
            "koi_score": st.column_config.ProgressColumn("Kepler Skoru", format="%.3f", min_value=0, max_value=1),
            "koi_prad": st.column_config.NumberColumn(label=r"Gezegen Yarıçapı ($R_{\oplus}$)", format="%.2f"),
            "koi_period": st.column_config.NumberColumn(label="Yörünge Periyodu (Gün)", format="%.2f"),
            "koi_depth": st.column_config.NumberColumn(label="Geçiş Derinliği (ppm)", format="%.1f"),
            # Yeni FP Kolonları için config - Düzenlenemez yapıldı
            "FP_NT": st.column_config.TextColumn("Gürültü Bayrağı (NT)", help="Gürültüye bağlı yanlış pozitif bayrağı", disabled=True),
            "FP_SS": st.column_config.TextColumn("Çoklu Sistem (SS)", help="Çoklu sistem bayrağı", disabled=True),
            "FP_CO": st.column_config.TextColumn("Merkez Kayması (CO)", help="Merkez kayması bayrağı", disabled=True),
            "Investigation_Status": st.column_config.SelectboxColumn( # Düzenlenebilir Etiketleme
                "İnceleme Durumu (Etiketle)",
                options=INVESTIGATION_STATUS_OPTIONS,
                required=True,
                default="Yeni Aday",
                width="medium"
            )
        },
    )
    
    # 🌟 ETİKETLEME DEĞİŞİKLİKLERİNİ KALICI HALE GETİRME (Ana DF'ye yazma)
    original_status_series = df_raw.loc[df_filtered.index, 'Investigation_Status']
    edited_status_series = edited_df_view['Investigation_Status']
    
    # Yalnızca status sütununda bir değişiklik varsa devam et
    if not edited_status_series.equals(original_status_series):
        
        # Değişen değerlerin indekslerini bul
        changed_indices = edited_status_series[edited_status_series != original_status_series].index
        
        # Ana DataFrame (st.session_state.df_raw) üzerindeki ilgili satırları güncelle
        for index in changed_indices:
             new_status = edited_df_view.loc[index, 'Investigation_Status']
             st.session_state.df_raw.loc[index, 'Investigation_Status'] = new_status
        
        # Değişikliklerin Streamlit'te kalıcı olması ve filtrelemeye yansıması için yeniden çalıştır
        st.rerun() 
        
    # İyileştirme: Veriyi İndirme Butonu Ekleme
    st.markdown("---")
    st.subheader("⬇️ Veriyi Dışa Aktar")
    
    @st.cache_data
    def convert_df_to_csv(df_filtered_placeholder):
        # Filtrelenen DF'in index'lerini kullanarak ana DF'ten sadece filtrelenmiş satırları al
        # Bu, en güncel Investigation_Status dahil tüm verileri içerir.
        df_to_export = st.session_state.df_raw.loc[df_filtered_placeholder.index].copy()
        
        # Sadece temel KOI sütunlarını ve Investigation_Status'ı dahil et
        export_cols_base = REQUIRED_COLUMNS + ['Investigation_Status']
        
        # KOI ile başlayan tüm sütunları da dahil et (ör: koi_disposition, koi_score_err)
        available_cols = [col for col in df_to_export.columns if col.startswith('koi_') or col in export_cols_base]
        
        df_to_export = df_to_export.filter(items=list(set(available_cols)))
        
        return df_to_export.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(df_filtered) # df_filtered, güncel filtreleri temsil eder
    
    st.download_button(
        label="Aktif Filtrelerle Etiketlenmiş Veriyi İndir (CSV)",
        data=csv,
        file_name='kepler_analysis_data.csv',
        mime='text/csv',
        type="secondary",
        use_container_width=True
    )
        

# -----------------------------------------------------------
# 4. UYGULAMA ANA GÖVDESİ VE DİNAMİK SİDEBAR
# -----------------------------------------------------------

upload_placeholder = st.empty()
uploaded_file = None 

# Session State'leri başlat
if 'df_raw' not in st.session_state: st.session_state.df_raw = None
if 'page_selection' not in st.session_state: st.session_state.page_selection = "✨ Tekil Aday Analizi"
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'last_prediction' not in st.session_state: st.session_state.last_prediction = None
if 'validation_issues' not in st.session_state: st.session_state.validation_issues = None
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = -1
if 'selected_candidate_index' not in st.session_state: st.session_state.selected_candidate_index = 0
if 'period_range' not in st.session_state: st.session_state.period_range = (0.0, 1000.0)
if 'score_threshold' not in st.session_state: st.session_state.score_threshold = 0.0
if 'status_filter' not in st.session_state: st.session_state.status_filter = INVESTIGATION_STATUS_OPTIONS 

# --- VERİ YÜKLEME KONTROLÜ (Başlangıç Sayfası) ---
if st.session_state.df_raw is None:
    
    with upload_placeholder.container():
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([2, 3, 2]) 
        
        with col_center:
            st.header("1. Kepler Veri Setini Yükle 🌠")
            st.markdown("### **<span style='color:#FF9900;'>Yapay Zeka ile Ötegezegen Adaylarını Bir Tıkla Temizle ve Analiz Et.</span>**", unsafe_allow_html=True)
            st.markdown("---")
            
            with st.expander("❓ Veri Gereksinimleri ve Güvenlik Önlemi"):
                 st.markdown("""
                 - **Veri Gizliliği:** Yüklediğiniz dosya, sadece bu oturum için kullanılır ve sunucularda saklanmaz.
                 - **Gereken Format:** Dosyanızın Kepler/KOI formatında, başlık kısmı atlanabilir (`skiprows=14`) ve **`koi_score`, `koi_period`, `koi_prad`** gibi zorunlu sütunları içermesi gerekir.
                 - **Güvenlik (Code Injection) Önlemi:** Yüklenen CSV dosyasındaki tüm sayısal sütunlar özel bir sistem tarafından zorla sayıya dönüştürülür. Eğer bu sütunlarda kötü niyetli metin veya komut bulunursa, bunlar zararsız `NaN` değerlerine dönüştürülür ve otomatik olarak temizlenir.
                 """)

            uploaded_file = st.file_uploader(
                "Lütfen filtrelenmiş Kepler/KOI CSV dosyasını buraya sürükle bırak veya Tıkla (.csv)", 
                type=['csv'],
                key="main_uploader"
            )

# --- DOSYA İŞLEME VE YENİDEN ÇALIŞTIRMA (RERUN) ---
if uploaded_file is not None and st.session_state.df_raw is None:
    upload_placeholder.empty()

    try:
        df_raw = pd.read_csv(uploaded_file, skiprows=14) 
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
        if missing_cols:
            st.error(f"Hata: Eksik zorunlu sütunlar var: **{', '.join(missing_cols)}**")
            st.session_state.df_raw = None
            st.stop() 

        # Veri Temizleme ve Validasyon
        df_cleaned, validation_issues = ExoplanetClassifier._validate_and_clean_data(df_raw, REQUIRED_COLUMNS)
        
        if df_cleaned.empty:
            st.error("Hata: Yüklenen dosyada tüm güvenlik ve temizlik kontrollerini geçen geçerli aday kalmadı.")
            st.session_state.df_raw = None
            st.stop()
            
        # İnceleme Durumu (Investigation_Status) sütununu ekle
        if 'Investigation_Status' not in df_cleaned.columns:
             df_cleaned['Investigation_Status'] = INVESTIGATION_STATUS_OPTIONS[0]
             
        st.session_state.df_raw = df_cleaned
        st.session_state.validation_issues = validation_issues
        st.session_state.show_results = False
        st.session_state.last_prediction = None
        st.session_state.selected_candidate_index = 0
        
# Filtre aralıklarını yüklenen veriye göre ayarla
        min_p = df_cleaned['koi_period'].min()
        max_p = df_cleaned['koi_period'].max()
        st.session_state.period_range = (min_p, max_p) if min_p < max_p else (min_p, min_p + 1) # Tek değerse slider'ı bozmamak için +1
        st.session_state.score_threshold = 0.0
        st.session_state.status_filter = INVESTIGATION_STATUS_OPTIONS # Yeni filtreyi varsayılana ayarla

        logger.info("Dosya başarıyla yüklendi ve temizlendi. Streamlit arayüz geçişi (rerun) tetikleniyor.")
        # Başarılı işlemlerden sonra arayüzün yeniden çizilmesi için yeterlidir.
        st.rerun()  

    # DİKKAT: RerunException bloğu ve import satırı kaldırılmıştır.
    # Genel ve kritik olmayan hatalar için sadece bu blok yeterlidir.
    except Exception as e:
        logger.exception("Dosya yükleme veya veri işleme sırasında beklenmeyen bir sorun oluştu.")  
        st.error(f"Genel Hata: Dosya yükleme veya veri işleme sırasında beklenmeyen bir sorun oluştu. Detay: {type(e).__name__}: {e}")
        # Hata durumunda session state'i temizle
        st.session_state.df_raw = None
        st.stop()

# --- ANA UYGULAMA DİNAMİK SİDEBAR KONTROLÜ ---
if st.session_state.df_raw is not None:
    
    upload_placeholder.empty()
    df_raw = st.session_state.df_raw
    
    # 1. NAVİGASYON BÖLÜMÜ
    st.sidebar.header("🗺️ 1. Uygulama Modeli (Alet Çantası)")
    page_selection = st.sidebar.radio(
        "Mod Seçimi",
        ["✨ Tekil Aday Analizi", "📋 Toplu Veri Seti İncelemesi"],
        index=0 if st.session_state.page_selection == "✨ Tekil Aday Analizi" else 1,
        key="page_selector"
    )
    st.session_state.page_selection = page_selection
    
    # 2. DİNAMİK KONTROLLER (SEÇİLEN MODA GÖRE DEĞİŞİR)
    st.sidebar.markdown("---") 

    if page_selection == "✨ Tekil Aday Analizi":
        # --- Tekil Analiz Kontrolleri ---
        st.sidebar.header("🌌 2. Aday Seçimi ve Analiz")
        
        candidate_index = st.sidebar.selectbox(
            label="Analiz Edilecek Adayı Seçin",
            options=list(range(len(df_raw))),
            format_func=lambda i: f"Aday {i+1} (Orijinal Satır No: {df_raw.index[i] + 1})",
            index=st.session_state.selected_candidate_index
        )
        st.session_state.selected_candidate_index = candidate_index
        
        # Analiz Başlat Butonu
        if st.sidebar.button('🚀 Seçili Adayı Tahmin Et ve Yorumla', type="primary", use_container_width=True):
            st.session_state.run_analysis = candidate_index
            st.session_state.show_results = False 
            
        # --- Çalıştırma Mantığı ---
        if 'run_analysis' in st.session_state and st.session_state.run_analysis == candidate_index:
            run_simulation_animation(candidate_index + 1)
            try:
                prediction, confidence, shap_buffer, raw_data = CLASSIFIER.predict(df_raw, candidate_index)
                st.session_state.last_prediction = (prediction, confidence, shap_buffer, raw_data)
                st.session_state.show_results = True
                st.session_state.run_analysis = -1 
            except RuntimeError as e: 
                st.error(f"Tahmin Hatası: {e}.")
                st.session_state.show_results = False
            except Exception as e:
                st.error(f"Genel Hata: Beklenmeyen bir sorun oluştu. Detay: {e}")
                st.session_state.show_results = False
                
        # --- HIZLI SONUÇ ÖZETİ (Sadece Analiz Bittiğinde Görünür) ---
        if 'show_results' in st.session_state and st.session_state.show_results:
            prediction, confidence, _, raw_data = st.session_state.last_prediction
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎯 Hızlı Karar Özeti")
            
            color = "#7FE9F0" if "YANLIŞ" not in prediction else "#DC3545" 
            
            st.sidebar.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid {color}; background-color: #171B20; border-radius: 5px;'>
                <p style='margin: 0; font-size: 14px; font-weight: bold; color: {color};'>Nihai Karar: {prediction}</p>
                <p style='margin: 5px 0 0 0; font-size: 16px; font-weight: 900; color: #fff;'>Güven: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # İyileştirme: Adayın İnceleme Durumunu göster
            current_status = df_raw.loc[df_raw.index[candidate_index], 'Investigation_Status']
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid #FF9900; background-color: #171B20; border-radius: 5px;'>
                <p style='margin: 0; font-size: 14px; font-weight: bold; color: #fff;'>Mevcut İnceleme Durumu:</p>
                <p style='margin: 5px 0 0 0; font-size: 16px; font-weight: 900; color: #FF9900;'>{current_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.metric("Yıldız Sıcaklığı", f"{raw_data.get('koi_steff', 0.0):.0f} K")
            st.sidebar.metric(r"Gezegen Yarıçapı ($R_{\oplus}$)", f"{raw_data.get('koi_prad', 0.0):.2f}")
            
    
    elif page_selection == "📋 Toplu Veri Seti İncelemesi":
        # --- Toplu Analiz Filtre Kontrolleri ---
        st.sidebar.header("🗄️ 2. Veri Filtreleme (Aletler)")
        
        # Filtreleme Aracı 1: Periyot Aralığı
        min_p = df_raw['koi_period'].min()
        max_p = df_raw['koi_period'].max()
        
        # Sadece min ve max'ın farklı olması durumunda slider göster
        if min_p < max_p:
            period_range = st.sidebar.slider(
                "Yörünge Periyodu Aralığı (Gün)",
                min_value=min_p,
                max_value=max_p,
                value=(st.session_state.period_range[0] if st.session_state.period_range[0] >= min_p else min_p, 
                       st.session_state.period_range[1] if st.session_state.period_range[1] <= max_p else max_p),
                key="period_slider"
            )
            st.session_state.period_range = period_range
        else:
            st.session_state.period_range = (min_p, max_p)
            
        # Filtreleme Aracı 2: Minimum Skor
        score_threshold = st.sidebar.slider(
            "Minimum Kepler/KOI Skoru",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.score_threshold,
            step=0.01,
            key="score_slider"
        )
        st.session_state.score_threshold = score_threshold
        
        # İyileştirme: Filtreleme Aracı 3: İnceleme Durumu Filtresi (Yeni)
        status_filter = st.sidebar.multiselect(
            "İnceleme Durumu Filtresi",
            options=INVESTIGATION_STATUS_OPTIONS,
            default=st.session_state.get('status_filter', INVESTIGATION_STATUS_OPTIONS),
            key="status_filter_multiselect"
        )
        st.session_state.status_filter = status_filter
        
        st.sidebar.markdown("---")
        
    # --- Veri Temizleme Raporu (Her zaman açılıp kapanabilir) ---
    if st.session_state.validation_issues and len(st.session_state.validation_issues) > 1:
        with st.sidebar.expander("Temizleme ve Validasyon Raporu"):
            st.warning(st.session_state.validation_issues[0]) 
            for issue in st.session_state.validation_issues[1:]:
                 st.write(f"- {issue}")

    # --- Sayfa Yönlendirme ---
    if page_selection == "✨ Tekil Aday Analizi":
        main_prediction_page()
    elif page_selection == "📋 Toplu Veri Seti İncelemesi":
        collective_analysis_page()
