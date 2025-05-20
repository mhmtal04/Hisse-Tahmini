import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

# Sayfa ayarları
st.set_page_config(page_title="Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Yarınki Fiyat ve Yüzde Tahmini (Olasılıklar Denklemi ile)")

# Kullanıcıdan hisse kodu al
symbol_input = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")
if symbol_input:
    symbol = symbol_input.upper()
    if not symbol.endswith(".IS"):
        symbol += ".IS"

    # Tarih aralığı
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
    with col2:
        end_date = st.date_input("Bitiş tarihi", datetime.date.today())

    # Veri çek
    st.write(f"**{symbol}** verisi indiriliyor...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty or len(data) < 20:
        st.warning("Veri alınamadı veya yetersiz. Daha geçerli bir hisse kodu ya da tarih aralığı seçin.")
    else:
        # Grafik
        st.line_chart(data["Close"].dropna(), use_container_width=True)

        # Gerçek zamanlı fiyat
        ticker = yf.Ticker(symbol)
        try:
            current_price = float(ticker.info.get("currentPrice", data["Close"].iloc[-1]))
        except:
            current_price = data["Close"].iloc[-1]
        st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")

        # Teknik göstergeler
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["Target"] = data["Close"].shift(-1)
        data = data.dropna()

        # Model eğitimi
        features = ["Close", "MA5", "MA10"]
        X = data[features]
        y = data["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.success(f"Model Ortalama Hata: ±{mae:.2f} TL")

        # Son verilerle tahmin
        latest_data = X.tail(1)
        prediction_raw = model.predict(latest_data)[0]

        # Olasılıklar denklemi (volatilite etkisi)
        try:
            recent_diff = data["Close"].iloc[-1] - data["Close"].iloc[-2]
            volatility = data["Close"].pct_change().rolling(window=5).std().iloc[-1] * 100
            volatility_value = float(volatility)
            olasilik_katsayisi = min(max(volatility_value / 5, -1), 1)
        except Exception:
            olasilik_katsayisi = 0
            recent_diff = 0

        prediction_adjusted = prediction_raw + recent_diff * olasilik_katsayisi

        # %10 fiyat limiti uygulaması
        upper_limit = current_price * 1.10
        lower_limit = current_price * 0.90
        predicted_price = max(min(prediction_adjusted, upper_limit), lower_limit)

        percent_change = ((predicted_price - current_price) / current_price) * 100
        percent_change = max(min(percent_change, 10), -10)

        # Sonuçları göster
        st.subheader("Tahmin Sonucu (Olasılıklar Denklemi ile):")
        st.write(f"Yarınki tahmini kapanış fiyatı: **{predicted_price:.2f} TL**")
        if abs(percent_change) >= 9.9:
            st.warning("Tahmin %10 BIST sınırına ulaştı.")
        st.write(f"Beklenen değişim: **{percent_change:+.2f}%**")

        with st.expander("Olasılıklar Denklemi Nedir?"):
            st.markdown("""
            Bu tahmin modeli, sadece makine öğrenmesi değil,
            aynı zamanda geçmiş fiyat değişimleri ve volatiliteye göre
            olasılık bazlı bir düzeltme uygular. Yani modelin tahmini,
            son hareketin yönüne ve piyasanın oynaklığına göre
            dinamik olarak ayarlanır.
            """)
