import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Gelişmiş Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 HİSSE TAHMİN BOTU")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
with col2:
    end_date = st.date_input("Bitiş tarihi", datetime.date.today())

if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"**{symbol}** verisi indiriliyor...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli bir hisse kodu veya tarih aralığı girin.")
    else:
        ticker = yf.Ticker(symbol)
        try:
            current_price = float(ticker.info["currentPrice"])
            st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")
        except:
            st.warning("Gerçek zamanlı fiyat alınamadı.")
            current_price = float(data["Close"].iloc[-1])

        # Teknik göstergeler
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()

        # Gerçek zamanlı fiyatı her satıra ekle (sabit, anlık fiyat)
        data["RealTimePrice"] = current_price

        # Tahmin hedefi: bir sonraki gün kapanışı
        data["Target"] = data["Close"].shift(-1)
        data = data.dropna()

        if data.shape[0] < 20:
            st.warning("Yeterli veri yok. Daha uzun zaman dilimi seçin.")
        else:
            features = ["Close", "MA5", "MA10", "RealTimePrice"]
            X = data[features]
            y = data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            st.success(f"Model Ortalama Hata: ±{mae:.2f} TL")

            latest_two = X.tail(2)

            # Tahmin ham değerleri
            today_pred_raw = model.predict(latest_two.iloc[[0]])[0]
            tomorrow_pred_raw = model.predict(latest_two.iloc[[1]])[0]

            # Son kapanış fiyatı farkı
            recent_diff = float(data["Close"].iloc[-1] - data["Close"].iloc[-2])

            # 5 günlük volatilite (yüzde)
            volatility = float(data["Close"].pct_change().rolling(window=5).std().iloc[-1] * 100)
            volatility_factor = min(max(volatility / 5, -1), 1)

            # Tahminleri volatiliteye göre ayarla
            today_pred = float(today_pred_raw) + recent_diff * volatility_factor
            tomorrow_pred = float(tomorrow_pred_raw) + recent_diff * volatility_factor

            # %10 limitler içinde düzelt
            upper_limit = current_price * 1.10
            lower_limit = current_price * 0.90

            today_pred = max(min(today_pred, upper_limit), lower_limit)
            tomorrow_pred = max(min(tomorrow_pred, upper_limit), lower_limit)

            st.subheader("Tahmin Sonuçları:")
            st.write(f"Bugünün kapanış fiyatı tahmini: **{today_pred:.2f} TL**")
            st.write(f"Yarınki kapanış fiyatı tahmini: **{tomorrow_pred:.2f} TL**")
