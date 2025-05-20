import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Gelişmiş Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Bugün ve Yarın için Kapanış Fiyatı Tahmini (Olasılıklar Denklemi ile)")

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
            current_price = data["Close"].iloc[-1]

        # Teknik göstergeler
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()

        # Volatilite: son 5 günlük yüzde değişim standart sapması (yüzde olarak)
        data["Pct_Change"] = data["Close"].pct_change()
        volatility = data["Pct_Change"].rolling(window=5).std() * 100

        # Gerçek zamanlı fiyatı her satıra ekle (sabit)
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

            # Son iki satırdan bugünün ve yarının kapanışını tahmin et
            latest_two = X.tail(2)
            today_pred_raw = model.predict(latest_two.iloc[[0]])[0]
            tomorrow_pred_raw = model.predict(latest_two.iloc[[1]])[0]

            # Volatilite katsayısı (olasılıklar denklemi için)
            recent_diff = data["Close"].iloc[-1] - data["Close"].iloc[-2]
            recent_volatility = volatility.iloc[-1]
            volatility_factor = min(max(recent_volatility / 5, -1), 1)

            # Tahminleri volatiliteye göre ayarla
            today_pred = today_pred_raw + recent_diff * volatility_factor
            tomorrow_pred = tomorrow_pred_raw + recent_diff * volatility_factor

            # %10 limitler içinde düzelt
            upper_limit = current_price * 1.10
            lower_limit = current_price * 0.90

            today_pred = max(min(today_pred, upper_limit), lower_limit)
            tomorrow_pred = max(min(tomorrow_pred, upper_limit), lower_limit)

            # Yüzde değişim hesapla
            today_pct_change = ((today_pred - current_price) / current_price) * 100
            tomorrow_pct_change = ((tomorrow_pred - current_price) / current_price) * 100

            st.subheader("Tahmin Sonuçları (Olasılıklar Denklemi ile):")
            st.write(f"Bugünün kapanış fiyatı tahmini: **{today_pred:.2f} TL** ({today_pct_change:+.2f}%)")
            st.write(f"Yarınki kapanış fiyatı tahmini: **{tomorrow_pred:.2f} TL** ({tomorrow_pct_change:+.2f}%)")

            if abs(today_pct_change) >= 9.9 or abs(tomorrow_pct_change) >= 9.9:
                st.warning("Tahmin %10 BIST sınırına ulaştı.")

            with st.expander("Olasılıklar Denklemi Nedir?"):
                st.markdown("""
                Bu tahmin modeli yalnızca makine öğrenmesiyle değil,
                aynı zamanda geçmiş fiyat hareketleri ve volatiliteye göre
                tahmini akıllı şekilde düzeltir. Böylece piyasanın oynaklığına
                göre tahmin dinamik olarak uyarlanır.
                """)
