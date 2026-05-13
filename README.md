# 🧠 Bilişsel Performans Skoru Tahmini
### Google Yapay Zeka ve Teknoloji Akademisi — YZTA 2026 Datathon

[![Kaggle](https://img.shields.io/badge/Kaggle-YZTA%202026%20Datathon-blue)](https://www.kaggle.com/competitions/yzta-2026-datathon)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![Score](https://img.shields.io/badge/Public%20Score-1.20577-orange)]()

---

## 📌 Problem Tanımı

Bu yarışmada katılımcıların uyku düzeni, yaşam alışkanlıkları ve günlük durumlarına ait değişkenler kullanılarak **bilişsel performans skoru** tahmin edilmektedir. Problem bir regresyon problemidir; hedef değişken sürekli sayısal bir değerdir.

---

## 📁 Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.csv` | Eğitim verisi — özellikler ve hedef değişken |
| `test_x.csv` | Test verisi — sadece özellikler |
| `sample_submission.csv` | Gönderim formatı örneği |
| `datathon-2025.ipynb` | Ana çalışma notebook'u |

---

## 🔍 Yaklaşımımız

### 1. Keşifsel Veri Analizi (EDA)
- Hedef değişken dağılımı incelendi
- Sayısal değişkenler ile hedef arasındaki korelasyonlar hesaplandı
- Kategorik değişkenlerin (ruh sağlığı, meslek) hedef üzerindeki etkisi görselleştirildi
- Eksik değerler tespit edildi

**Temel bulgular:**
- `stres_skoru` en güçlü negatif korelasyon (-0.587)
- `rem_yuzdesi` en güçlü pozitif korelasyon (+0.443)
- Ruh sağlığı durumu skoru güçlü şekilde etkiliyor (Sağlıklı: 6.49, Anksiyete+Depresyon: 3.59)
- Emekliler en yüksek (7.94), Avukatlar en düşük (4.72) ortalama skora sahip

### 2. Veri Temizleme
- Kategorik tutarsızlıklar düzeltildi (Spain → Ispanya, Lawyer → Avukat vb.)
- Aşırı uç değerler temizlendi

### 3. Feature Engineering
EDA bulgularından yola çıkarak 50 anlamlı özellik türetildi:

| Özellik | Açıklama |
|---|---|
| `kaliteli_uyku` | REM + derin uyku yüzdesi toplamı |
| `uyku_verimi` | Kaliteli uyku / (uyanma sayısı + 1) |
| `dalma_uyanma` | Uykuya dalma süresi × uyanma sayısı |
| `stres_rem` | Stres × düşük REM etkileşimi |
| `stres_calisma` | Stres × çalışma saati |
| `aktivite_skoru` | Günlük adım / çalışma saati |
| `gece_uyari` | Kafein + ekran süresi (kötü gece alışkanlıkları) |
| `ruh_sagligi_encoded` | Ruh sağlığı durumu sıralı sayısal kodlama |
| `meslek_encoded` | Meslek → ortalama performans skoru |

### 4. Model Eğitimi
3 farklı gradient boosting algoritması kullanıldı:

| Model | CV RMSE |
|---|---|
| **CatBoost** | 1.2169 ± 0.0108 |
| XGBoost | 1.2203 ± 0.0110 |
| LightGBM | 1.2224 ± 0.0107 |

**Ensemble:** `0.50 × CatBoost + 0.25 × LightGBM + 0.25 × XGBoost`

### 5. Cross Validation
- StratifiedKFold (10-Fold) kullanıldı
- Hedef değişken `pd.qcut` ile 10 gruba bölünerek stratifikasyon sağlandı
- Her fold'da tutarlı sonuçlar elde edildi

### 6. Optimizasyon
- **Optuna** ile 3 model için hiperparametre optimizasyonu yapıldı (30 deneme)
- Gereksiz featurelar `feature_importance` analizine göre budandı

---

## 📊 Sonuçlar

| Versiyon | Public Score | Açıklama |
|---|---|---|
| Version 1 | 1.23367 | Temel model |
| Version 2 | 1.20808 | Outlier kaldırıldı + Optuna |
| Version 3 | 1.20577 | StratifiedKFold + feature budama |

---

## 🛠️ Kullanılan Kütüphaneler

```python
pandas, numpy, matplotlib, seaborn
scikit-learn
lightgbm
xgboost
catboost
optuna
```

---

## 👥 Takım

**Datathon Takım 105**
- Rukiye Dişçi
- Levent Kök
- Doğa Alışkan
