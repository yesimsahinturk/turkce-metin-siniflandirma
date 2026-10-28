# Türkçe Metin Sınıflandırma Sistemi

Bu proje, Türkçe kısa metinleri konu kategorilerine ayırmak için TF-IDF ve Logistic Regression kullanan başlangıç düzeyinde bir doğal dil işleme (NLP) uygulamasıdır.

## Kategoriler
- Teknoloji
- Eğitim
- Spor
- Sağlık

> Not: `data/metinler.csv` dosyasındaki örnek veri kümesi eğitim ve demonstrasyon amacıyla hazırlanmış küçük bir veri kümesidir. Gerçek kullanım veya akademik performans iddiası için veri kümesi büyütülmeli, çeşitlendirilmeli ve ayrı test verileriyle değerlendirilmelidir.

## Kullanılan teknolojiler
- Python
- pandas
- scikit-learn
- TF-IDF (metni sayısal özelliklere dönüştürme)
- Logistic Regression (sınıflandırma modeli)

## Kurulum

Python 3.10 veya üzeri önerilir.

```bash
pip install -r requirements.txt
```

## Modeli eğitme ve değerlendirme

```bash
python train.py
```

Bu komut veri kümesini eğitim/test olarak ayırır, TF-IDF + Logistic Regression modelini eğitir ve doğruluk ile sınıflandırma raporunu terminalde gösterir.

## Yeni metin tahmini

```bash
python predict.py
```

Açılan komut satırına Türkçe bir metin yaz. Program tahmin edilen kategoriyi gösterir. Çıkmak için `q` yaz.

## Proje yapısı

```text
turkce_metin_siniflandirma/
├── data/
│   └── metinler.csv
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Geliştirme fikirleri
- Daha büyük ve dengeli bir Türkçe veri kümesi kullanmak
- Farklı modelleri (Naive Bayes, Linear SVM) karşılaştırmak
- Confusion matrix görselleştirmesi eklemek
- Streamlit ile web arayüzü oluşturmak

## CV'de nasıl belirtilebilir?
**Türkçe Metin Sınıflandırma Sistemi | Python, scikit-learn, TF-IDF**
- Türkçe metinleri konu kategorilerine ayırmak için TF-IDF ve Logistic Regression tabanlı bir sınıflandırma modeli geliştirdim.
- Eğitim/test ayrımı ve sınıflandırma raporuyla model performansını değerlendirdim.

CV'ne eklemeden önce projeyi kendi bilgisayarında çalıştır, sonuçları incele ve kodun nasıl çalıştığını anladığından emin ol.
