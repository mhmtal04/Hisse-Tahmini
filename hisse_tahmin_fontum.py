import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

# XGBoost'u isteğe bağlı yükleme denemesi
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgboost_available = False

# LSTM için
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Yarınki Fiyat Tahmini (RandomForest, LSTM ve Opsiyonel XGBoost)")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

# Tarih aralığı seçimi
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
with col2:
    end_date = st.date_input("Bitiş tarihi", datetime.date.today() + datetime.timedelta(days=1))

# Model seçenekleri
models = ["RandomForest", "LSTM"]
if xgboost_available:
    models.insert(1, "XGBoost")  # XGBoost varsa ekle

model_option = st.selectbox("Model seçin:", models)

if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"**{symbol}** verisi indiriliyor...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli bir hisse kodu veya tarih aralığı girin.")
    else:
        # Gerçek zamanlı fiyat al
        ticker = yf.Ticker(symbol)
        try:
            current_price = float(ticker.info["currentPrice"])
            st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")
        except:
            st.warning("Gerçek zamanlı fiyat alınamadı, son kapanış fiyatı kullanılacak.")
            current_price = data["Close"].dropna().iloc[-1]

        # Teknik göstergeler
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["Target"] = data["Close"].shift(-1)
        data = data.dropna()

        if data.shape[0] < 20:
            st.warning("Yeterli veri yok. Daha uzun zaman dilimi seçin.")
        else:
            if model_option in ["RandomForest", "XGBoost"]:
                features = ["Close", "MA5", "MA10"]
                X = data[features]
                y = data["Target"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                if model_option == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(random_state=42)
                else:
                    # XGBoost seçildi, kullanıma hazır
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                st.success(f"{model_option} Model Ortalama Hata: ±{mae:.2f} TL")

                latest_data = X.tail(1)
                prediction_raw = float(model.predict(latest_data)[0])

            else:  # LSTM modeli
                scaler = MinMaxScaler()
                scaled_close = scaler.fit_transform(data[["Close"]])

                TIME_STEPS = 5
                X_lstm = []
                y_lstm = []

                for i in range(len(scaled_close) - TIME_STEPS - 1):
                    X_lstm.append(scaled_close[i:(i + TIME_STEPS), 0])
                    y_lstm.append(scaled_close[i + TIME_STEPS, 0])

                X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

                split = int(0.8 * len(X_lstm))
                X_train, X_test = X_lstm[:split], X_lstm[split:]
                y_train, y_test = y_lstm[:split], y_lstm[split:]

                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(TIME_STEPS, 1)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mae')

                model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

                preds = model.predict(X_test)
                preds_rescaled = scaler.inverse_transform(preds)
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
                st.success(f"LSTM Model Ortalama Hata: ±{mae:.2f} TL")

                last_sequence = scaled_close[-TIME_STEPS:].reshape((1, TIME_STEPS, 1))
                prediction_raw_scaled = model.predict(last_sequence)[0][0]
                prediction_raw = scaler.inverse_transform([[prediction_raw_scaled]])[0][0]

            # Olasılıklar Denklemi ile düzeltme
            recent_diff = data["Close"].iloc[-1] - data["Close"].iloc[-2]
            volatility = data["Close"].pct_change().rolling(window=5).std().iloc[-1] * 100
            volatility_value = float(volatility)
            olasilik_katsayisi = min(max(volatility_value / 5, -1), 1)
            prediction_adjusted = float(prediction_raw + recent_diff * olasilik_katsayisi)

            # %10 limit kontrolü
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
                Bu tahmin modeli yalnızca makine öğrenmesiyle değil,
                aynı zamanda geçmiş fiyat hareketleri ve volatiliteye göre
                tahmini akıllı şekilde düzeltir. Böylece piyasanın oynaklığına
                göre tahmin dinamik olarak uyarlanır.
                """)
