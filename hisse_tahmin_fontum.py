
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Yarınki Fiyat ve Yüzde Tahmini (Olasılıklar Denklemi ile)")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

# Tarih aralığı seçimi
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
with col2:
    end_date = st.date_input("Bitiş tarihi", datetime.date.today() + datetime.timedelta(days=1))

if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"**{symbol}** verisi indiriliyor...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli bir hisse kodu veya tarih aralığı girin.")
    else:
        # Gerçek zamanlı fiyat
        ticker = yf.Ticker(symbol)
        try:
            current_price = float(ticker.info["currentPrice"])
            st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")
        except:
            st.warning("Gerçek zamanlı fiyat alınamadı.")
            current_price = data["Close"].dropna().iloc[-1]

        # Kapanış fiyatı grafiği
        st.line_chart(data["Close"], use_container_width=True)

        # Özellikleri hazırla
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["Target"] = data["Close"].shift(-1)
        data = data.dropna()

        if data.shape[0] < 20:
            st.warning("Yeterli veri yok. Daha uzun zaman dilimi seçin.")
        else:
            features = ["Close", "MA5", "MA10"]
            X = data[features]
            y = data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            st.success(f"Model Ortalama Hata: ±{mae:.2f} TL")

            latest_data = X.tail(1)
            prediction_raw = model.predict(latest_data)[0]

            # Olasılıklar denklemi etkisi (volatiliteye dayalı katsayı)
            recent_diff = data["Close"].iloc[-1] - data["Close"].iloc[-2]
volatility = data["Close"].pct_change().rolling(window=5).std().iloc[-1] * 100
volatility_value = float(volatility)
olasilik_katsayisi = min(max(volatility_value / 5, -1), 1)
prediction_adjusted = prediction_raw + recent_diff * olasilik_katsayisi

# Tahmini fiyatı %10 limitlere göre düzelt
            upper_limit = current_price * 1.10
            lower_limit = current_price * 0.90
            predicted_price = max(min(prediction_adjusted, upper_limit), lower_limit)

            percent_change = ((predicted_price - current_price) / current_price) * 100
            percent_change = max(min(percent_change, 10), -10)

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
