# Pnömoni Tespit Sistemi (Pneumonia Detection System)

Web tabanlı yapay zeka destekli akciğer röntgeni analiz sistemi.

## 🎯 Özellikler

- **Yapay Zeka Analizi**: MobileNetV2 tabanlı derin öğrenme modeli (%90-95 doğruluk)
- **Grad-CAM Görselleştirme**: AI kararlarının görsel açıklaması
- **Hasta Yönetimi**: TC Kimlik No ile hasta kayıt sistemi
- **PDF Raporlama**: Detaylı tarama raporları
- **İstatistik Paneli**: Gerçek zamanlı analiz ve grafikler
- **Arama ve Filtreleme**: Gelişmiş kayıt yönetimi

## 🚀 Kurulum

### Gereksinimler

```bash
Python 3.8+
pip
```

### Adımlar

1. Depoyu klonlayın:
```bash
git clone [REPO_URL]
cd X-ray
```

2. Sanal ortam oluşturun:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Modeli indirin:
- `best_model.h5` dosyasını proje kök dizinine yerleştirin

5. Uygulamayı başlatın:
```bash
cd X-ray
python app.py
```

6. Tarayıcıda açın: `http://localhost:5000`

## 📁 Proje Yapısı

```
X-ray/
├── app.py                 # Ana Flask uygulaması
├── database.py            # SQLite veritabanı işlemleri
├── gradcam.py            # Grad-CAM görselleştirme
├── pdf_generator.py      # PDF rapor oluşturma
├── train_model.py        # Model eğitim scripti
├── DL.py                 # Alternatif CNN modeli
├── requirements.txt      # Python bağımlılıkları
├── best_model.h5         # Eğitilmiş AI modeli (eklenecek)
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── scripts.js
│   ├── uploads/          # Yüklenen röntgenler
│   ├── heatmaps/         # Grad-CAM görselleri
│   └── reports/          # Oluşturulan PDF'ler
└── templates/
    ├── index.html        # Ana sayfa
    ├── result.html       # Sonuç sayfası
    ├── history.html      # Kayıt geçmişi
    └── dashboard.html    # İstatistik paneli
```

## 🔬 Teknolojiler

- **Backend**: Flask, TensorFlow, Keras, SQLite
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **AI Model**: MobileNetV2 (Transfer Learning)
- **Visualization**: Grad-CAM, OpenCV
- **Reports**: ReportLab

## 📊 Model Performansı

- **Doğruluk (Accuracy)**: ~90-95%
- **Model**: MobileNetV2 (ImageNet pre-trained)
- **Input Size**: 224x224x3
- **Dataset**: Chest X-Ray Images (Pneumonia)

## 🎨 Özellikler Detayı

### 1. Tarama Analizi
- Röntgen görüntüsü yükleme
- Anlık AI analizi
- Güven skoru hesaplama
- Grad-CAM ısı haritası

### 2. Hasta Yönetimi
- TC Kimlik No ile kayıt
- Hasta bilgileri (ad, yaş, cinsiyet, telefon)
- Tarama geçmişi

### 3. Raporlama
- Profesyonel PDF raporları
- Görüntüler ve ısı haritaları
- Tıbbi tavsiyeler
- Yasal uyarılar

### 4. Dashboard
- Toplam tarama sayısı
- Pnömoni/Normal dağılımı
- Zaman bazlı grafikler
- Son taramalar

## ⚠️ Önemli Notlar

- Bu sistem **eğitim ve araştırma** amaçlıdır
- Profesyonel tıbbi teşhis için kullanılmamalıdır
- Kesin tanı için **uzman hekime** danışılmalıdır
- Model sonuçları %100 doğru değildir

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.



---

**Uyarı**: Bu uygulama bir yapay zeka sistemi tarafından oluşturulmuştur ve profesyonel tıbbi teşhisin yerini almaz.
