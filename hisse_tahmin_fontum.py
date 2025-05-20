import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Hisse Tahmin Botu", layout="centered")
st.title("📊 Hisse Tahmin Botu")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

# Son 3 aylık tarih aralığı
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=90)

if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"Veri indiriliyor: {symbol} ({start_date} - {end_date})")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli bir hisse kodu girin.")
    else:
        ticker = yf.Ticker(symbol)
        try:
            current_price = float(ticker.info["currentPrice"])
            st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")
        except:
            st.warning("Gerçek zamanlı fiyat alınamadı.")
            current_price = data["Close"].iloc[-1]

        # Teknik göstergeler
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["RealTimePrice"] = current_price
        data["Target"] = data["Close"].shift(-1)
        data = data.dropna()

        if len(data) < 20:
            st.warning("Yeterli veri yok. Daha uzun tarih aralığı seçin.")
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

            # Tahmin: bugün ve yarın
            latest_two = X.tail(2).copy()
            today_pred_raw = model.predict([latest_two.iloc[0]])[0]
            tomorrow_pred_raw = model.predict([latest_two.iloc[1]])[0]

            # Olasılıklar denklemi düzeltmesi
            if len(data) >= 6:
                recent_diff = data["Close"].iloc[-1] - data["Close"].iloc[-2]
                volatility = data["Close"].pct_change().rolling(window=5).std().iloc[-1] * 100
                katsayi = min(max(volatility / 5, -1), 1)

                today_pred = today_pred_raw + recent_diff * katsayi
                tomorrow_pred = tomorrow_pred_raw + recent_diff * katsayi
            else:
                today_pred = today_pred_raw
                tomorrow_pred = tomorrow_pred_raw

            # BIST limiti (%10)
            upper_limit = current_price * 1.10
            lower_limit = current_price * 0.90

            today_pred = float(max(min(today_pred, upper_limit), lower_limit))
            tomorrow_pred = float(max(min(tomorrow_pred, upper_limit), lower_limit))

            st.subheader("Tahmin Sonuçları:")
            st.write(f"Bugün için kapanış tahmini: **{today_pred:.2f} TL**")
            st.write(f"Yarın için kapanış tahmini: **{tomorrow_pred:.2f} TL**")

            with st.expander("Model Açıklaması"):
                st.markdown("""
                - **Model:** Random Forest (100 ağaç)
                - **Girdi:** Son fiyatlar, hareketli ortalamalar, gerçek zamanlı fiyat
                - **Çıkış:** Kapanış tahmini
                - **Düzenleme:** Volatilite ve son gün farkına göre ayarlama (olasılıklar denklemi)
                """)
