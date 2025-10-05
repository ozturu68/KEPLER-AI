## 🇹🇷 Türkçe

# 🔭 Kepler-AI: Yüksek Güvenilirlikli Ötegezegen Keşif Asistanı

<img width="2000" height="2000" alt="logo (1)" src="https://github.com/user-attachments/assets/7530512b-7ab0-4cc0-9ce5-c0f2f7780001" />


## 🌟 Proje Özeti

KEPLER-AI, NASA'nın **Kepler Uzay Teleskobu** verilerinde yer alan potansiyel ötegezegen adaylarını (KOI) analiz etmek için geliştirilmiş, **Streamlit** tabanlı interaktif bir platformdur.

Bu uygulama, yüksek doğruluklu **CatBoost Sınıflandırma Modeli** ile adayları sınıflandırmakla kalmaz, aynı zamanda **SHAP (SHapley Additive exPlanations)** kullanarak modelin karar mekanizmasını şeffaf bir şekilde açıklar. Kullanıcılar, kendi KOI veri setlerini yükleyebilir, adayları toplu olarak etiketleyebilir ve bilimsel keşif süreçlerini hızlandırabilir.

### Model Performansı

| Metrik | Değer | Not |
| :--- | :--- | :--- |
| **Model Tipi** | CatBoost Sınıflandırıcısı | Yüksek Hız ve Güvenilirlik |
| **Tahmin Metodu** | Monte Carlo Simülasyonu | Sağlam ve Güvenilir Tahmin Skoru |
| **Yorumlanabilirlik** | SHAP (XAI) | Karar şeffaflığı için kritik |

## 🚀 Canlı Uygulama (Live Demo)

Uygulama yayına alındıktan sonra bu başlık altında canlı link yer alacaktır:

➡️ https://kepler-ai.streamlit.app/

## ⚙️ Uygulama Özellikleri

- **Güvenli Veri Yükleme:** Kepler/KOI formatında CSV dosyalarını yükleme, güvenlik ve temizlik kontrolleri.
- **Tekil Aday Analizi:** Seçilen bir aday için model tahmini, güven skoru ve temel parametrelerin gösterimi.
- **Model Yorumlanabilirliği (XAI):** Tahminleri açıklayan ve karara etki eden özellikleri gösteren **SHAP Waterfall Grafikleri**.
- **Toplu İnceleme:** Filtreleme özellikli interaktif tablo (`st.data_editor`) üzerinden adayları toplu etiketleme.
- **Veri Bütünlüğü:** Streamlit Session State kullanarak toplu etiketlemelerin ana veri setine kalıcı olarak yazılması.

## 📁 Proje Dizini Yapısı

| Dizin/Dosya | Açıklama |
| :--- | :--- |
| `app.py` | Streamlit Web Uygulaması ve Tüm Analiz/UI Mantığı. |
| `models/` | Kaydedilmiş Model (`CatBoost`), Skaler (`Scikit-learn`) ve Özellik İsimleri (`joblib` ile yüklü). |
| `requirements.txt` | Tüm Python Bağımlılıkları ve Versiyonları (CatBoost, SHAP dahil). |
| `README.md` | Proje Tanıtım ve Kılavuz Sayfası. |

## 🛠️ Kurulum ve Yerel Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için:

1. **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KullaniciAdiniz/Kepler-AI-Proje.git](https://github.com/KullaniciAdiniz/Kepler-AI-Proje.git)
    cd Kepler-AI-Proje
    ```

2. **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```
    *(Uygulama otomatik olarak tarayıcınızda açılacaktır.)*

---

## 🇬🇧 English 


# 🔭 Kepler-AI: High-Confidence Exoplanet Discovery Assistant

![Streamlit Application Screenshot or Logo](Insert a striking screenshot/GIF of your application here)

## 🌟 Project Overview

KEPLER-AI is an interactive **Streamlit** platform developed to analyze potential exoplanet candidates (KOIs) found in the data from NASA's **Kepler Space Telescope**.

This application not only classifies candidates using a high-accuracy **CatBoost Classification Model** but also utilizes **SHAP (SHapley Additive exPlanations)** to transparently explain the model's decision-making process. Users can upload their own KOI datasets, collectively label candidates, and accelerate the scientific discovery process.

### Model Performance

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Model Type** | CatBoost Classifier | High Speed and Reliability |
| **Prediction Method** | Monte Carlo Simulation | Robust and High-Confidence Prediction Score |
| **Interpretability** | SHAP (XAI) | Critical for decision transparency |

## 🚀 Live Application (Live Demo)

The live link will be placed under this heading once the application is deployed:

➡️ https://kepler-ai.streamlit.app/

## ⚙️ Application Features

- **Secure Data Upload:** Upload Kepler/KOI format CSV files, complete with security and data cleaning checks.
- **Single Candidate Analysis:** Model prediction, confidence score, and display of key parameters for a selected candidate.
- **Model Interpretability (XAI):** **SHAP Waterfall Plots** to explain predictions and show feature contributions to the decision.
- **Collective Review:** Batch labeling of candidates (`Confirmed`, `False Positive`, etc.) via an interactive, filterable table (`st.data_editor`).
- **Data Integrity:** Persistent saving of collective labels back to the main dataset using Streamlit's Session State.

## 📁 Project Directory Structure

| Directory/File | Description |
| :--- | :--- |
| `app.py` | Streamlit Web Application and all Analysis/UI Logic. |
| `models/` | Saved Machine Learning Assets: Model (`CatBoost`), Scaler (`Scikit-learn`), and Feature Names (loaded with `joblib`). |
| `requirements.txt` | All Python Dependencies and Versions (including CatBoost, SHAP). |
| `README.md` | Project Introduction and Guide Page. |

## 🛠️ Setup and Local Run

To run the project on your local machine (after cloning from GitHub):

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/Kepler-AI-Proje.git
    cd Kepler-AI-Proje
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the Application:**
    ```bash
    streamlit run app.py
    ```
    *(The application will automatically open in your browser.)*
