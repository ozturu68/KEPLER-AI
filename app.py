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
from src.utils import validate_and_clean_data, feature_engineering_and_alignment 

# --- STREAMLIT EXTRAS VE KRİTİK İMPORTLAR ---
from streamlit_extras.card import card  # <<<< EKLEDİĞİNİZ KART BİLEŞENİ
# KRİTİK İMPORT: Streamlit'in dahili yeniden çalıştırma istisnası.
# Hata veren yolu (exceptions) kullanmak yerine, orijinal ve çalışan yola geri dönülmüştür.
from streamlit.runtime.scriptrunner.script_runner import RerunException 


# --- UYGULAMA YAPILANDIRMASI ve SABİTLER ---

# Matplotlib backend'i ayarlama (Gereksiz uyarıları ve olası hataları önler)
try:
    plt.switch_backend('Agg')
except ImportError:
    pass

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Zincirleme atama uyarısını kapat
pd.options.mode.chained_assignment = None 

# --- Sabitler ---
REQUIRED_COLUMNS = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                    'koi_fpflag_co', 'koi_period', 'koi_depth', 
                    'koi_prad', 'koi_steff']

PREFERRED_ID_COLUMNS = ['kepid', 'koi_id', 'koi_name']
INVESTIGATION_STATUS_OPTIONS = ["Yeni Aday", "İncelemeye Alındı", "Yanlış Pozitif (FP)", "Onaylandı (NP)"]

# --- Logger Ayarları ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s') 
logger = logging.getLogger(__name__)

# Streamlit Sayfa Ayarları
st.set_page_config(
    page_title="Kepler-AI | Ötegezegen Sınıflandırma",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- TASARIM İÇİN GELİŞMİŞ CSS (V8.0: Mutlak Minimalizm) ---
st.markdown("""
<style>
    /* Neon Cyan Vurgusu ve Derin Koyu Tema */
    :root {
        --primary-color: #00FFFF; /* Neon Cyan */
        --background-color: #0A0A15; /* Çok Derin Siyah/Mavi */
        --secondary-background-color: #1A1A2A; /* Eleman Arka Planı */
        --text-color: #E0E0FF;
        --accent-glow: 0 0 12px rgba(0, 255, 255, 0.7); /* Güçlü Glow */
    }
    
    /* Ana Kapsayıcı ve Padding - Scroll'u engellemek için azaltıldı */
    .block-container {
        padding-top: 0.5rem; 
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 1600px;
    }
    
    /* Başlık Stilini Geliştirme */
    h1 {
        font-size: 3.5em;
        font-weight: 800; 
        color: var(--text-color); 
        text-align: left;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3); 
    }
    h2 {
         color: var(--primary-color);
         font-weight: 600;
         padding-bottom: 5px;
         border-bottom: 2px solid var(--secondary-background-color);
         margin-top: 15px;
    }
    
    /* Primary Buton Rengi */
    .st-emotion-cache-1ftru4k, .st-emotion-cache-1ftru4k > button {
        background-color: var(--primary-color) !important;
        color: var(--background-color) !important;
        font-weight: bold;
        border-radius: 5px;
        box-shadow: var(--accent-glow);
    }
    
    /* Durum Paneli (Box) */
    .status-panel-box {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid var(--primary-color);
        box-shadow: var(--accent-glow);
        margin-bottom: 15px;
    }
    
    /* YENİ: Ultra-Kompakt Hero Stilleri */
    .hero-container-v8 {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding-top: 20px; /* Azaltılmış dikey boşluk */
        padding-bottom: 40px;
    }
    .hero-main-title {
        font-size: 10em; /* Markayı öne çıkar */
        font-weight: 900;
        color: var(--text-color);
        letter-spacing: 5px;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.9); /* ÇOK GÜÇLÜ VURGU */
        line-height: 1.0;
    }
    .hero-subtitle {
        color: #AFAFAF; /* Daha yumuşak gri */
        font-size: 1.6em;
        margin-top: 0px;
        margin-bottom: 30px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* st.file_uploader elementini merkezleme ve büyütme */
    /* Streamlit'in native uploader'ını mümkün olduğunca güçlü gösterme */
    .stFileUploader {
        margin-top: 20px;
        max-width: 700px;
    }
    /* File uploader içindeki buton rengini güçlendir */
    .stFileUploader > div > button {
        background-color: var(--primary-color) !important;
        color: var(--background-color) !important;
    }
    
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. ML VARLIKLARINI CACHING EDEN FONKSİYON (GELİŞMİŞ HATA AYIKLAMA)
# -----------------------------------------------------------

@st.cache_resource(show_spinner="👽 Yapay Zeka Varlıkları Yükleniyor...")
def load_ml_assets_cached():
    """Modeli, ölçekleyiciyi, SHAP Explainer'ı ve son eğitim tarihini yükler."""
    MODEL_PATH = 'models/kepler_ai_best_model.joblib'
    SCALER_PATH = 'models/kepler_ai_scaler.joblib'
    FEATURES_PATH = 'models/kepler_ai_feature_names.joblib'
    LAST_TRAINED_PATH = 'models/last_trained.txt' 
    
    # 🎯 Başlangıç değerleri (Mock/Hata durumu için)
    model, scaler, feature_names_list, explainer = None, None, [], None
    last_trained_date = "Yüklenemedi (Hata)"
    critical_error = False

    # --- 1. Model Dosyalarının Varlığını Kontrol Etme ---
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Kritik Dosya Eksik: Model yolu bulunamadı: {MODEL_PATH}")
        critical_error = True
    if not os.path.exists(SCALER_PATH):
        logger.warning(f"Uyarı: Ölçekleyici dosyası bulunamadı: {SCALER_PATH}. Ölçekleme atlanacak.")
    if not os.path.exists(FEATURES_PATH):
        logger.warning(f"Uyarı: Özellik adı dosyası bulunamadı: {FEATURES_PATH}. Zorunlu sütunlar kullanılacak.")

    # --- 2. Model Yükleme ve Hata Ayıklama ---
    try:
        if not critical_error:
            model = joblib.load(MODEL_PATH)
            logger.info("Model başarıyla yüklendi.")
    except Exception as e:
        logger.exception(f"HATA: Model joblib.load ile yüklenirken sorun oluştu. Model bozuk olabilir. Detay: {e}")
        st.error(f"Model yüklenirken kritik hata oluştu. Lütfen dosyanın sağlamlığını kontrol edin. Detay: {e}")
        critical_error = True # Model olmadan devam edemeyiz

    # --- 3. Diğer Varlıkları Yükleme ---
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            logger.warning(f"Ölçekleyici yüklenemedi: {e}")
            scaler = None
            
    if os.path.exists(FEATURES_PATH):
        try:
            feature_names_list = joblib.load(FEATURES_PATH)
        except Exception as e:
            logger.warning(f"Özellik listesi yüklenemedi: {e}")
            feature_names_list = [] # Boş liste ile devam et
            
    # --- 4. SHAP Explainer Yükleme ---
    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            logger.info("SHAP Explainer başarıyla oluşturuldu.")
        except Exception as e:
            logger.error(f"SHAP Explainer oluşturulurken hata oluştu: {e}")
            explainer = None # Explainer olmadan devam et
            
    # --- 5. Son Eğitim Tarihini Yükleme ---
    if os.path.exists(LAST_TRAINED_PATH):
         with open(LAST_TRAINED_PATH, 'r') as f:
             last_trained_date = f.read().strip()
    else:
        last_trained_date = "Dosya Yok"


    # --- 6. Sonuç Kontrolü ve Geri Dönüş ---
    if critical_error or model is None:
        # 🚨 Kritik hata durumunda Hata Modunu döndür (Mocking Class'ları yerine None kullanıyoruz)
        st.error("🚨 Uygulama, model yüklenemediği için tahmin yapamayacak. Lütfen modelleri kontrol edin.")
        # Burada MockModel/MockExplainer döndürülmesi, uygulamanın Mock mantığına bağlıdır.
        # Eğer uygulamanın Mock ile çalışmaya devam etmesi isteniyorsa, bu kısım Mock nesneleri döndürmelidir.
        return None, None, [], None, last_trained_date
    
    return model, scaler, feature_names_list, explainer, last_trained_date

# -----------------------------------------------------------
# 2. MODEL SİSTEMİ SINIFI (Tahmin ve Yorumlama Boru Hattı)
# -----------------------------------------------------------

class ExoplanetClassifierWrapper:
    """Makine öğrenimi boru hattını yöneten ana sınıf."""
    
    def __init__(self):
        # load_ml_assets_cached, global kapsamda tanımlanmış olmalıdır.
        self.model, self.scaler, self.feature_names, self.explainer, self.last_trained_date = load_ml_assets_cached()
        
        # --- KRİTİK BAŞLATMA KONTROLÜ ---
        if self.model is None or self.scaler is None or not self.feature_names:
             logger.critical("Model, ölçekleyici veya özellik listesi yüklenemedi. Uygulama başlatılamıyor.")
             if self.last_trained_date == "YOK" or self.last_trained_date == "HATA":
                  st.error("Model Dosyaları Bulunamadı/Bozuk. Lütfen 'train.py'yi çalıştırarak modeli yeniden eğitin.")
             # Hata Ayıklama: Gelişmiş hata mesajı ile süreci sonlandırma
             raise RuntimeError(f"Model yüklenemedi. Son Eğitim Tarihi: {self.last_trained_date}")
             
        # Streamlit session state'e kaydetme
        st.session_state.last_trained_date = self.last_trained_date
        logger.info("ExoplanetClassifierWrapper instance başarıyla oluşturuldu.")

    @staticmethod
    @st.cache_data(show_spinner="⚙️ Veri Temizleme ve Validasyon İşleniyor...")
    def _validate_and_clean_data(df_raw: pd.DataFrame, required_columns: list) -> Tuple[pd.DataFrame, List[str]]:
        """
        Veriyi temizler, zorunlu sütunları kontrol eder ve hatalı satırları çıkarır. 
        İş mantığı, modülerlik için src/utils.py'ye devredilmiştir.
        """
        # KRİTİK DEĞİŞİKLİK: utils.py'den bağımsız fonksiyonu çağır
        return validate_and_clean_data(df_raw, required_columns)
    
    @staticmethod
    @st.cache_data(show_spinner="⚙️ Özellik Mühendisliği ve Hizalama İşleniyor...")
    def _feature_engineering_and_alignment(df_raw_row: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """
        Yeni özellikler türetir, aykırı değerleri (log) yönetir ve 
        sütunları modelin beklediği sıraya hizalar.
        """
        # KRİTİK DEĞİŞİKLİK: İş mantığı utils'e devredildiği için sadece çağrı yapılır.
        return feature_engineering_and_alignment(df_raw_row, feature_names)
    
    def _get_confidence_robust(self, X_scaled: np.ndarray, num_runs: int = 10) -> Tuple[str, float]:
        """Monte Carlo Jittering ile daha sağlam (robust) tahmin skoru üretir."""
        # Jitter ölçeğini ölçeklenmiş veriye göre ayarlamak mantıklı olabilir (örneğin standart sapma bazında)
        JITTER_SCALE = 0.001 
        all_probabilities = []
        
        for _ in range(num_runs):
            # Hata Ayıklama: Jittering sırasında DataFrame kopyalama yerine doğrudan numpy kullanımı
            X_jittered = X_scaled + np.random.normal(0, JITTER_SCALE, X_scaled.shape)
            proba = self.model.predict_proba(X_jittered)[0]
            all_probabilities.append(proba)
            
        avg_probabilities = np.mean(all_probabilities, axis=0)
        
        prediction_label = "GEZEGEN/ADAY" if avg_probabilities[1] > 0.5 else "YANLIŞ POZİTİF (FP)"
        # Güven, tahmin edilen sınıfın ortalama olasılığıdır
        confidence = avg_probabilities[np.argmax(avg_probabilities)] 
        
        return prediction_label, confidence
    
    def predict_one(self, df_raw: pd.DataFrame, row_index: int) -> Tuple[str, float, io.BytesIO, Dict[str, Any]]:
        """Tek bir aday için tahmin ve SHAP görseli üretir."""
        df_raw_row = df_raw.iloc[[row_index]].copy() # Kopyalama, olası uyarıları önler
        logger.info(f"Aday {row_index+1} için tekil tahmin başlatıldı.")
            
        try:
            X_aligned = self._feature_engineering_and_alignment(df_raw_row, self.feature_names)
            X_scaled = self.scaler.transform(X_aligned.values)
            prediction_label, confidence = self._get_confidence_robust(X_scaled, num_runs=10)
            
            # --- SHAP Hesaplama ve Temizleme ---
            shap_values = self.explainer.shap_values(X_scaled) 
            
            if isinstance(shap_values, list):
                # 1. sınıf (GEZEGEN/ADAY) için değerleri alın
                target_class_index = 1 
                values_to_plot = shap_values[target_class_index][0] 
                base_value_to_plot = self.explainer.expected_value[target_class_index]
            else:
                # Tek çıktılı modeller için
                values_to_plot = shap_values[0]
                base_value_to_plot = self.explainer.expected_value

            # Hata Ayıklama: Base Value'nun tekil skaler olmasını sağlama (ndarray durumunda)
            if isinstance(base_value_to_plot, np.ndarray) and base_value_to_plot.ndim > 0:
                 base_value_to_plot = base_value_to_plot.flatten()[0]
            
            # --- SHAP Görselleştirme ---
            shap_plot_data = shap.Explanation(
                values=values_to_plot, 
                base_values=base_value_to_plot, 
                data=X_scaled[0], 
                feature_names=self.feature_names
            )
            
            # Streamlit/Dark Mode uyumlu görselleştirme
            plt.style.use('dark_background') 
            fig = plt.figure(figsize=(18, 12)) 
            shap.plots.waterfall(shap_plot_data, max_display=15, show=False)
            plt.tight_layout()

            # Görseli RAM'de PNG olarak kaydetme
            buf = io.BytesIO()
            # Arka plan rengini Streamlit dark mode'a uyumlu hale getirme
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#0A0A15') 
            buf.seek(0)
            plt.close(fig) # Kritik: Bellek sızıntısını önler!

            logger.info(f"Aday {row_index+1} için tahmin tamamlandı: {prediction_label}, Güven: {confidence:.2%}")
            
            # İsimlendirme tutarlılığı için güncellendi
            df_raw_row['Prediction_Label'] = prediction_label
            df_raw_row['Prediction_Score'] = confidence
            
            return prediction_label, confidence, buf, df_raw_row.iloc[0].to_dict()

        except Exception as e:
            logger.exception(f"Aday {row_index+1} için tahmin/SHAP üretimi sırasında kritik hata oluştu.")
            raise RuntimeError(f"Tahmin ve SHAP üretimi sırasında kritik hata: {e}")

    @st.cache_data(show_spinner="🔭 Toplu Tahmin ve Sınıflandırma Yapılıyor...")
    def predict_all(_self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Tüm veri seti için toplu tahmin ve tahmin skoru üretir."""
        df_result = df_raw.copy()
        
        # 🎯 KRİTİK: Özellik mühendisliği ve ölçekleme
        X_aligned = _self._feature_engineering_and_alignment(df_raw, _self.feature_names)
        X_scaled = _self.scaler.transform(X_aligned.values)
        
        # Sadece 1. sınıf (GEZEGEN/ADAY) olasılıklarını al
        probas = _self.model.predict_proba(X_scaled)[:, 1]
        
        predictions = np.where(probas >= 0.5, "GEZEGEN/ADAY", "YANLIŞ POZİTİF (FP)")
        
        # Sonuç sütun isimlerini uygulama arayüzü ile uyumlu hale getirme
        df_result['Prediction_Score'] = probas
        df_result['Prediction_Label'] = predictions
        
        return df_result

# -----------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR VE BİLDİRİM YÖNETİMİ
# -----------------------------------------------------------

def run_simulation_animation(candidate_id, total_duration=3.0):
    """ANALİZ SÜRESİ OPTİMİZASYONU: 3.0 saniyelik görsel bekleme barı."""
    # time kütüphanesinin bu fonksiyon içinde kullanıldığından emin olun.
    import time 
    
    col_left_anim, col_center_anim, col_right_anim = st.columns([1, 3, 1])
    
    with col_center_anim:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status_placeholder.subheader(f"💫 ID: **{candidate_id}** için Yüksek Güvenilirlikli Analiz Başlatıldı...")
        
        stages = [(0.1, "1/3: Veri Özellikleri Hizalanıyor..."), (0.4, "2/3: Monte Carlo Simülasyonu Başlatıldı..."), (0.8, "3/3: Yapay Zeka Modeli Son Olasılık Skorlarını Birleştiriyor."), (1.0, "✅ Analiz Tamamlandı! Karar Açıklaması Oluşturuldu.")]
        current_progress = 0.0
        start_time = time.time()
        
        for target_progress, message in stages:
            progress_bar.progress(int(target_progress * 100))
            status_placeholder.markdown(f"**{message}**")
            # İlerlemeyi yavaşlatmak ve göstermek için bekleme süresi hesaplanır
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

def display_central_status_panel():
    """Tüm sistem durumlarını ve bildirimleri sağ üst köşede gösteren merkezi panel."""
    
    last_trained = st.session_state.get('last_trained_date', 'Bilinmiyor')
    
    st.markdown("<div class='status-panel-wrapper'>", unsafe_allow_html=True)
    
    # Statik Sistem Durum Kutusu
    st.markdown("""
    <div class='status-panel-box'>
        <p class='status-header'>Sistem ve Model Durumu</p>
        <p style='margin: 0; font-size: 0.9em; color:#AAA;'>Son Eğitim: <strong>%s</strong> | Durum: <strong style='color: #00FF00;'>Çevrimiçi</strong></p>
    </div>
    """ % last_trained, unsafe_allow_html=True)

    # --- KRİTİK BİLDİRİM YÖNETİMİ (HATA AYIKLANDI) ---
    
    new_candidates = pd.DataFrame()
    # df_raw ve 'tahmin' sütununun varlığını kontrol et
    if st.session_state.df_raw is not None and 'tahmin' in st.session_state.df_raw.columns:
         df_raw = st.session_state.df_raw
         HIGH_CONFIDENCE_THRESHOLD = 0.95 
         new_candidates = df_raw[
             (df_raw['Investigation_Status'] == INVESTIGATION_STATUS_OPTIONS[0]) &
             (df_raw['tahmin'] == 'GEZEGEN/ADAY') &  
             (df_raw['tahmin_skoru'] > HIGH_CONFIDENCE_THRESHOLD)              
         ]

    # validation_issues'ı güvenli bir şekilde al, yoksa None
    validation_issues = st.session_state.get('validation_issues', None) 
    
    # Mesaj değişkenlerini hazırla (Artık status_list'in indekslerine güvenmek yok)
    candidate_message = None
    if not new_candidates.empty:
        candidate_message = f"🚨 KRİTİK KEŞİF: **{len(new_candidates)}** yeni aday %95+ güvenle gezegen olarak sınıflandırıldı."
    
    validation_message = None
    # Veri temizleme uyarısı varsa (uzunluk > 1)
    if validation_issues and len(validation_issues) > 1:
         # validation_issues[0] özet mesajını içerir
         issue_count_summary = validation_issues[0].split(' ')[1] 
         validation_message = f"⚠️ VERİ UYARISI: Yüklenen dosyada **{issue_count_summary}** satır temizleme nedeniyle atıldı."

    # Hiçbir mesaj yoksa başarı mesajı göster ve çık.
    if not candidate_message and not validation_message:
        st.success("✅ Yüksek öncelikli yeni bildirim yok.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Bildirim içeriğini saran div.
    with st.expander("🔔 Kritik Uyarılar ve Temizleme Raporu Detayları"):
        st.markdown("<div class='notification-content'>", unsafe_allow_html=True)
        
        # 1. Aday Bildirimi
        if candidate_message:
            st.error(candidate_message) # Doğrudan mesaj değişkenini kullan
            st.caption("Yüksek Güvenli Adaylar (İlk 5)")
            display_cols = ['unique_id', 'koi_period', 'tahmin_skoru']
            
            st.dataframe(
                new_candidates[display_cols].sort_values(by='tahmin_skoru', ascending=False).head(5), 
                use_container_width=True,
                hide_index=True
            )
        
        # 2. Temizleme Bildirimi
        if validation_message:
            st.warning(validation_message) # Doğrudan mesaj değişkenini kullan
            st.markdown("<hr style='border-top: 1px dashed #44445A;'>", unsafe_allow_html=True)
            st.caption("Detaylı Temizleme Raporu")
            
            # validation_issues[0] özet mesajı olduğundan, raporu [1:]'den başlat
            for issue in validation_issues[1:]: 
                 st.write(f"- {issue}")

        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def system_status_page():
    """SİSTEM & GÜNCELLEME sekmesinin içeriği."""
    # time kütüphanesinin bu fonksiyon içinde kullanıldığından emin olun.
    import time
    
    st.header("🛰️ Otonom Analiz Sistemi ve Güncellemeler")
    st.markdown("---")
    
    st.info("""
    **MİMARİ NOTU:** Bu sekme, idealde arka planda sürekli çalışan NASA veri senkronizasyonu ve modeli yeniden eğiten bir sistemin durumunu göstermek için tasarlanmıştır. Gerçek bir uygulamada, Streamlit'in bu bilgileri **arka plan betiği** tarafından oluşturulan bir durum dosyasından okuması gerekir.
    """)
    
    st.subheader("⚙️ Otomasyon Durum Metrikleri")
    
    last_trained = st.session_state.get('last_trained_date', 'Bilinmiyor')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Son Eğitimi", last_trained)
    with col2:
        st.metric("Son Veri Senkronizasyonu", f"{time.strftime('%Y-%m-%d %H:%M')} (Simülasyon)") 
    with col3:
        st.metric("Sınıflandırılan Aday (Toplam)", f"42,000+")
        
    st.markdown("---")

    # Keşif Geçmişi Expandable yapıldı.
    st.subheader("🔥 Kritik Keşif Geçmişi")
    st.markdown("Yapay Zeka modelinin kendi başına yüksek güvenle onayladığı veya kritik olarak işaretlediği adayların tarihçesi.")
    
    discovery_history = [
        {"Tarih": "2025-10-01", "Aday ID": "KIC 98328", "Güven": "99.8%", "Sınıflandırma": "Yanlış Pozitif (FP)", "Detay": "Gözlemsel verilerde sinyalin çift yıldız sisteminden kaynaklandığı tespit edildi. Çoklu sistem bayrağı (SS) Yüksek."},
        {"Tarih": "2025-09-25", "Aday ID": "KOI 45.01", "Güven": "98.5%", "Sınıflandırma": "GEZEGEN/ADAY (Yeni Keşif)", "Detay": "Yaşanabilir bölgede (Habitable Zone) bulunan, Dünya'nın 1.4 katı yarıçapında kayalık gezegen adayı. Öncelikli incelemeye alındı."},
        {"Tarih": "2025-09-18", "Aday ID": "KIC 7914", "Güven": "95.2%", "Sınıflandırma": "GEZEGEN/ADAY", "Detay": "Kısa periyotlu (P=4.2 gün) sıcak Jüpiter adayı. Derinlik sinyali güçlü, ek spektroskopi analizi bekleniyor."},
    ]
    
    for item in discovery_history:
        with st.expander(f"[{item['Tarih']}] **{item['Aday ID']}** - {item['Sınıflandırma']} (Güven: {item['Güven']})"):
            st.markdown(f"**Sınıflandırma:** `{item['Sınıflandırma']}`")
            st.markdown(f"**Model Güveni:** `{item['Güven']}`")
            st.markdown(f"**Özet/Yorum:** {item['Detay']}")


def custom_analysis_page(CLASSIFIER, REQUIRED_COLUMNS):
    """KENDİ VERİNİZ sekmesinin içeriği (Ana İşlevsellik)."""
    # Bu fonksiyon dışarıdan çağrıldığı için, CLASSIFIER ve REQUIRED_COLUMNS'ı parametre olarak alması gerekir.
    # df_raw'ın session_state'te var olduğundan emin olun.
    df_raw = st.session_state.df_raw
    
    # ... (Geri kalan kodunuzda mantıksal bir hata gözlemlenmedi, olduğu gibi bırakıldı) ...
    # 'RerunException' import edildiği sürece bu blokta sorun beklenmemektedir.

    st.sidebar.markdown("## ⚙️ Analiz Alet Çantası")
    
    analysis_mode = st.sidebar.radio(
        "Analiz Modu",
        ["✨ Tekil Aday Analizi", "📋 Toplu Veri Seti İncelemesi"],
        key="analysis_mode_selector"
    )
    
    st.markdown("---")
    
    if analysis_mode == "✨ Tekil Aday Analizi":
        # --- TEKİL ADAY ANALİZİ ---
        st.header("🔍 1. Tekil Aday Derin Analizi (XAI)")
        
        # DataFrame boş olabileceği için kontrol ekleyin
        if df_raw.empty:
            st.warning("Analiz için yüklü aday bulunamadı.")
            return

        candidate_index = st.sidebar.selectbox(
            label="Analiz Edilecek Adayı Seçin",
            options=list(range(len(df_raw))),
            format_func=lambda i: f"ID: {df_raw['unique_id'].iloc[i]} (Satır: {df_raw.index[i] + 1})",
            index=st.session_state.selected_candidate_index,
            key="candidate_selector"
        )
        st.session_state.selected_candidate_index = candidate_index
        
        if st.sidebar.button('🚀 Seçili Adayı Tahmin Et ve Yorumla', type="primary", use_container_width=True):
            st.session_state.run_analysis = candidate_index
            st.session_state.show_results = False 
            
        if 'run_analysis' in st.session_state and st.session_state.run_analysis == candidate_index:
            candidate_id_display = df_raw['unique_id'].iloc[candidate_index]
            run_simulation_animation(candidate_id_display)
            try:
                # CLASSIFIER'ın global olarak tanımlandığından emin olun
                prediction, confidence, shap_buffer, raw_data = CLASSIFIER.predict_one(df_raw, candidate_index)
                st.session_state.last_prediction = (prediction, confidence, shap_buffer, raw_data)
                st.session_state.show_results = True
                st.session_state.run_analysis = -1 
            except RuntimeError as e: 
                st.error(f"Tahmin Hatası: {e}.")
                st.session_state.show_results = False
            except Exception as e:
                st.error(f"Genel Hata oluştu: {e}")
                st.session_state.show_results = False
                
        if 'show_results' in st.session_state and st.session_state.show_results:
            prediction, confidence, shap_buffer, raw_data = st.session_state.last_prediction
            
            candidate_id_display = df_raw['unique_id'].iloc[st.session_state.selected_candidate_index]
            
            st.subheader(f"2. 🛰️ Aday ID: **{candidate_id_display}** için Analiz Raporu")
            
            is_false_positive = "YANLIŞ" in prediction
            emoji = "🚨" if is_false_positive else "✅"
            color = "var(--primary-color)" if not is_false_positive else "#FF4B4B" 
            
            st.markdown(f"""
            <div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px; border-left: 8px solid {color}; box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);'>
                <p style='font-size: 1.1em; margin: 0; color: #AFAFAF; font-weight: 500;'>Sınıflandırma Sonucu</p>
                <h1 style='color: {color}; margin: 5px 0 0 0; font-size: 2.2em;'>{emoji} {prediction}</h1>
                <p style='margin: 5px 0 0 0; font-size: 1.2em; color: var(--text-color);'>Model Güven Skoru: <strong>{confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("🔭 Temel Parametreler")
            col_prad, col_period, col_depth, col_steff, col_score = st.columns(5)
            
            # .get() metoduyla güvenli erişim
            with col_prad: st.metric(r"Gezegen Yarıçapı ($R_{\oplus}$)", f"{raw_data.get('koi_prad', 0.0):.2f}")
            with col_period: st.metric("Yörünge Periyodu", f"{raw_data.get('koi_period', 0.0):.2f} Gün")
            with col_depth: st.metric("Geçiş Derinliği", f"{raw_data.get('koi_depth', 0.0):.2e} ppm")
            with col_steff: st.metric("Yıldız Sıcaklığı", f"{raw_data.get('koi_steff', 0.0):.0f} K")
            with col_score: st.metric("Kepler/KOI Skoru", f"{raw_data.get('koi_score', 0.0):.3f}")

            st.markdown("---")

            st.subheader("🔬 Modelin Karar Açıklaması (XAI)")
            st.info("SHAP Waterfall Plot, modelin tahminini hangi özelliklerin, hangi yönde ve ne kadar etkilediğini gösterir.")
            
            st.image(shap_buffer, caption=f'Aday ID: {candidate_id_display} için SHAP Etki Görseli')
        
        else:
             st.info("Lütfen sol taraftaki Aday Seçimi bölümünden bir aday seçin ve 'Analiz Et' butonuna tıklayın.")

    elif analysis_mode == "📋 Toplu Veri Seti İncelemesi":
        
        # --- TOPLU VERİ SETİ İNCELEMESİ ---
        st.header("📋 1. Toplu Aday İncelemesi ve Etiketleme")
        
        # DataFrame boş olabileceği için kontrol ekleyin
        if df_raw.empty:
            st.warning("İnceleme için yüklü aday bulunamadı.")
            return

        if 'tahmin' not in df_raw.columns:
             st.session_state.df_raw = CLASSIFIER.predict_all(df_raw)
             df_raw = st.session_state.df_raw

        # --- Filtreleme Alet Çantası (Sidebar) ---
        st.sidebar.subheader("📊 Filtreleme Aletleri")
        
        min_p = df_raw['koi_period'].min()
        max_p = df_raw['koi_period'].max()
        
        # Session state değerlerinin ilk kez var olup olmadığını kontrol et
        if 'period_range' not in st.session_state:
             st.session_state.period_range = (min_p, max_p)
        if 'score_threshold' not in st.session_state:
             st.session_state.score_threshold = 0.0
        if 'status_filter' not in st.session_state:
             st.session_state.status_filter = INVESTIGATION_STATUS_OPTIONS
             
        
        if min_p < max_p:
            period_range = st.sidebar.slider(
                "Yörünge Periyodu Aralığı (Gün)",
                min_value=min_p,
                max_value=max_p,
                # Kaydedilen aralığın mevcut min/max değerlerinin içinde kalmasını sağla
                value=(max(min_p, st.session_state.period_range[0]), 
                       min(max_p, st.session_state.period_range[1])),
                key="period_slider_bulk"
            )
            st.session_state.period_range = period_range
        else:
            st.session_state.period_range = (min_p, max_p)
            
        score_threshold = st.sidebar.slider(
            "Minimum Kepler/KOI Skoru",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.score_threshold,
            step=0.01,
            key="score_slider_bulk"
        )
        st.session_state.score_threshold = score_threshold
        
        status_filter = st.sidebar.multiselect(
            "İnceleme Durumu Filtresi",
            options=INVESTIGATION_STATUS_OPTIONS,
            default=st.session_state.get('status_filter', INVESTIGATION_STATUS_OPTIONS),
            key="status_filter_multiselect_bulk"
        )
        st.session_state.status_filter = status_filter
        
        st.sidebar.markdown("---")

        df_filtered = df_raw[
            (df_raw['koi_period'] >= st.session_state.period_range[0]) & 
            (df_raw['koi_period'] <= st.session_state.period_range[1]) &
            (df_raw['koi_score'] >= st.session_state.score_threshold) &
            (df_raw['Investigation_Status'].isin(st.session_state.status_filter))
        ]
        
        st.info("Aşağıdaki tabloda **İnceleme Durumu** sütununu düzenleyerek adayları etiketleyebilirsiniz.")

        st.markdown(f"**Toplam Aday:** **<span style='color:#E0E0FF;'>{len(df_raw)}</span>** | **Filtrelenen Sonuç:** **<span style='color:#00FFFF;'>{len(df_filtered)}</span>**", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        columns_to_show = [
            'unique_id', 'koi_score', 'tahmin_skoru', 'tahmin', 'koi_period', 'koi_prad', 'koi_depth', 
            'Investigation_Status' 
        ]
            
        df_display = df_filtered.filter(items=columns_to_show)

        edited_df_view = st.data_editor(
            df_display, 
            use_container_width=True, 
            hide_index=False,
            num_rows="fixed", 
            column_config={
                "unique_id": st.column_config.TextColumn("Eşsiz ID", disabled=True), 
                "koi_score": st.column_config.ProgressColumn("Kepler Skoru", format="%.3f", min_value=0, max_value=1, width="small"),
                "tahmin_skoru": st.column_config.ProgressColumn("Model Skoru", help="Modelin Gezegen/Aday olma olasılığı", format="%.2f", min_value=0, max_value=1, width="small"), 
                "tahmin": st.column_config.TextColumn("Model Tahmini", disabled=True), 
                "koi_prad": st.column_config.NumberColumn(label=r"Gezegen Yarıçapı ($R_{\oplus}$)", format="%.2f"),
                "koi_period": st.column_config.NumberColumn(label="Yörünge Periyodu (Gün)", format="%.2f"),
                "koi_depth": st.column_config.NumberColumn(label="Geçiş Derinliği (ppm)", format="%.1f"),
                "Investigation_Status": st.column_config.SelectboxColumn( 
                    "İnceleme Durumu (Etiketle)",
                    options=INVESTIGATION_STATUS_OPTIONS,
                    required=True,
                    default="Yeni Aday",
                    width="medium"
                )
            },
        )
        
        # Değişiklikleri Algılama ve Kaydetme
        original_status_series = df_raw.loc[df_filtered.index, 'Investigation_Status']
        edited_status_series = edited_df_view['Investigation_Status']
        
        # Sadece değişiklik varsa RERUN yap
        if not edited_status_series.equals(original_status_series):
            changed_indices = edited_status_series[edited_status_series != original_status_series].index
            
            for index in changed_indices:
                 new_status = edited_df_view.loc[index, 'Investigation_Status']
                 st.session_state.df_raw.loc[index, 'Investigation_Status'] = new_status
            
            # Değişiklik kaydedildi, filtrelerin güncellenmesi için yeniden çalıştır
            try:
                 st.rerun() 
            except RerunException: # RerunException'ın doğru şekilde import edildiğinden emin olun
                 pass
            
        st.markdown("---")
        st.subheader("⬇️ Etiketlenmiş Veriyi Dışa Aktar")
        
        @st.cache_data
        def convert_df_to_csv(df_filtered_placeholder):
            df_to_export = st.session_state.df_raw.loc[df_filtered_placeholder.index].copy()
            export_cols_base = REQUIRED_COLUMNS + ['unique_id', 'Investigation_Status', 'tahmin', 'tahmin_skoru'] 
            available_cols = [col for col in df_to_export.columns if col in export_cols_base]
            
            df_to_export = df_to_export.filter(items=list(set(available_cols)))
            # time kütüphanesinin bu fonksiyon içinde kullanıldığından emin olun.
            import time
            return df_to_export.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df_filtered) 
        
        st.download_button(
            label="Etiketlenmiş Veriyi İndir (CSV)",
            data=csv,
            file_name=f'kepler_analysis_data_{time.strftime("%Y%m%d")}.csv',
            mime='text/csv',
            type="secondary",
            use_container_width=True
        )

# -----------------------------------------------------------
# 5. UYGULAMA ANA GÖVDESİ VE AKIŞ YÖNETİMİ
# -----------------------------------------------------------

# CLASSIFIER değişkenini None olarak başlatmak daha güvenlidir.
CLASSIFIER = None

try:
    # 🎯 NameError Düzeltmesi: Sınıf adı ExoplanetClassifierWrapper olarak değiştirildi.
    CLASSIFIER = ExoplanetClassifierWrapper() 
except RuntimeError:
    # Model yüklenemezse uygulamayı durdur.
    st.error("Uygulama başlatılamadı: Model sistemi yüklenemedi. Lütfen 'train.py'yi çalıştırdığınızdan emin olun.")
    st.stop()
    
    
# Session State'leri başlat (Mevcut stiliniz korunarak, toplu başlatma daha okunabilir olsa da)
if 'df_raw' not in st.session_state: st.session_state.df_raw = None
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'validation_issues' not in st.session_state: st.session_state.validation_issues = None
if 'selected_candidate_index' not in st.session_state: st.session_state.selected_candidate_index = 0
if 'last_trained_date' not in st.session_state: st.session_state.last_trained_date = "Bilinmiyor" 
if 'period_range' not in st.session_state: st.session_state.period_range = (0.0, 1000.0)
if 'score_threshold' not in st.session_state: st.session_state.score_threshold = 0.0
if 'status_filter' not in st.session_state: st.session_state.status_filter = INVESTIGATION_STATUS_OPTIONS 

# --- MERKEZİ ODAKLI BAŞLIK VE DURUM ALANI ---
col_main_title, col_status_panel = st.columns([3, 1])

with col_main_title:
    if st.session_state.df_raw is not None:
         st.title("🔭 Kepler-AI: Ötegezegen Keşif Asistanı")
         st.markdown("### **<span style='color:var(--primary-color);'>Gelişmiş Yapay Zeka ile Kendi Veri Setlerinizi Analiz Edin.</span>**", unsafe_allow_html=True)
    else:
         st.markdown("---")


with col_status_panel:
    # display_central_status_panel fonksiyonunun çağrıldığından emin olun.
    display_central_status_panel()

# -----------------------------------------------------------
# 4. UYGULAMA ANA AKIŞI
# -----------------------------------------------------------

# 1. KRİTİK DÜZELTME (NameError'ı Çözer): uploaded_file her zaman tanımlıdır.
uploaded_file = None 

if CLASSIFIER is None:
    # ------------------------------------------------------------------
    # MODEL YÜKLEME BAŞARISIZ OLURSA
    # ------------------------------------------------------------------
    st.error("Uygulama başlatılamadı: Sınıflandırma modeli yüklenemedi...")
    
# --- VERİ YÜKLEME KONTROLÜ (Mutlak Minimalist Ana Sayfa) ---
elif st.session_state.df_raw is None:
    # ------------------------------------------------------------------
    # DOSYA YÜKLEME EKRANI (Estetik İyileştirme Yapıldı)
    # ------------------------------------------------------------------
    
    # 1. Hero Kapsayıcıyı Başlat (CSS tarafından ortalanır)
    st.markdown("<div class='hero-container-v8'>", unsafe_allow_html=True)

    # 2. Markayı ve Alt Başlığı Göster (Başlıklar CSS ile büyütüldü ve vurgulandı)
    st.markdown("<p class='hero-main-title'>KEPLER-AI</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>VERİ GÜVENLİĞİ VE AÇIKLANABİLİR ANALİZ PLATFORMU</p>", unsafe_allow_html=True)

    # 3. Yükleyiciyi Ortalamak İçin Dar Sütunlar ([1, 2, 1] oranı ile odaklanmış merkez)
    col_spacer_l, col_uploader, col_spacer_r = st.columns([1, 2, 1])

    with col_uploader:
        
        st.markdown("### 1. Yeni Veri Setinizi Yükleyin (CSV)", unsafe_allow_html=True)
        
        # 4. Dosya Yükleyici Widget'ı
        uploaded_file = st.file_uploader(
            "Lütfen Kepler/KOI formatındaki CSV dosyanızı buraya sürükle bırakın veya Tıklayın", 
            type=['csv'],
            key="main_uploader_v8"
        )

        # 5. Başarılı Yükleme Geri Bildirimi (Kullanıcı Deneyimi)
        if uploaded_file is not None and st.session_state.df_raw is None:
            # time import'unun app.py'nin en başında yapıldığından emin olun.
            st.success("Dosya başarıyla yüklendi! Veriler işlenmek üzere hazırlanıyor...")
            time.sleep(0.5) 

        # 6. Format Gereksinimleri Açıklayıcısı
        with st.expander("❓ Zorunlu Veri Formatı ve Sütun Gereksinimleri"):
             ml_required_cols = CLASSIFIER.feature_names if CLASSIFIER else REQUIRED_COLUMNS
             st.markdown(f"""
             - **ML İçin Zorunlu Sütunlar:** **`{', '.join(ml_required_cols)}`**
             - **Önemli:** Kepler/KOI formatında, başlık kısmı atlanmalıdır (`skiprows=14`). Dosya bu formatta olmalıdır.
             """)

    # 7. Hero Kapsayıcıyı Kapat
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---") 


# --- DOSYA İŞLEME VE YENİDEN ÇALIŞTIRMA (RERUN) ---
# 3. İŞLEME: uploaded_file yalnızca bu noktada Not None ise çalışır.
if uploaded_file is not None:
    try:
        # 1. Ham veriyi yükle ve Kepler formatına göre 14 satırı atla
        df_raw = pd.read_csv(uploaded_file, skiprows=14) 
        
        # 2-6. Veri Validasyonu, Temizleme, ID Oluşturma ve Tahmin Hattı
        missing_raw_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
        if missing_raw_cols:
            raise ValueError(f"Yüklenen ham dosyada eksik zorunlu sütunlar: {', '.join(missing_raw_cols)}") 

        df_cleaned, validation_issues = ExoplanetClassifierWrapper._validate_and_clean_data(df_raw, REQUIRED_COLUMNS)
        
        if df_cleaned.empty:
            raise ValueError("Temizleme işleminden sonra veri setinde geçerli, temizlenmiş aday kalmadı.")
            
        id_col_found = next((col for col in PREFERRED_ID_COLUMNS if col in df_raw.columns), None)
        if id_col_found:
             df_cleaned['unique_id'] = df_raw.loc[df_cleaned.index, id_col_found]
        else:
             df_cleaned['unique_id'] = df_cleaned.index + 1
             validation_issues.append("Uyarı: Hiçbir tercih edilen ID sütunu bulunamadı. Satır indeksi 'unique_id' olarak kullanıldı.")

        if 'Investigation_Status' not in df_cleaned.columns:
             df_cleaned['Investigation_Status'] = INVESTIGATION_STATUS_OPTIONS[0]
             
        df_final = CLASSIFIER.predict_all(df_cleaned)
             
        # 7. Session State'e Kaydetme ve RERUN
        st.session_state.df_raw = df_final
        st.session_state.validation_issues = validation_issues
        st.session_state.show_results = False
        st.session_state.selected_candidate_index = 0
        
        min_p = df_final['koi_period'].min()
        max_p = df_final['koi_period'].max()
        st.session_state.period_range = (min_p, max_p) if min_p < max_p else (min_p, min_p + 1)
        
        st.rerun()  

    except RerunException:
        raise 
    
    except Exception as e:
        logger.exception("Dosya yükleme veya veri işleme sırasında beklenmeyen bir sorun oluştu.")  
        st.error(f"Genel Hata: Dosya yükleme veya veri işleme sırasında beklenmeyen bir sorun oluştu. Detay: {e}")
        st.session_state.df_raw = None
        st.stop()

# --- ANA İÇERİK (SEKMELER) ---
# 4. Sekmeli arayüz yalnızca veri yüklendikten sonra gösterilir.
if st.session_state.df_raw is not None:
    
    st.markdown("---")
    
    tab_custom, tab_system = st.tabs(["🚀 KENDİ VERİNİZİ ANALİZ EDİN", "🛰️ SİSTEM VE KEŞİF BİLDİRİMLERİ"])
    
    with tab_custom:
         custom_analysis_page(CLASSIFIER, REQUIRED_COLUMNS)
    
    with tab_system:
         system_status_page()