import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Hisse Tahmin Botu", layout="centered")
st.title("📊 HİSSE TAHMİN BOTU")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

# -------- Tarih aralığı (varsayılan 30 gün) --------
st.markdown("**Varsayılan: Son 30 gün. Değiştirmek istersen tarih seçim kutularını kullan.**")

use_custom_range = st.checkbox("Tarih aralığını kendim seçmek istiyorum")
today = datetime.date.today()
default_start = today - datetime.timedelta(days=30)

if use_custom_range:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Başlangıç", default_start)
    with col2:
        end_date = st.date_input("Bitiş", today)
else:
    start_date, end_date = default_start, today

# -------- Yeni matematik seçenekleri --------
st.sidebar.markdown("### 🔧 Gelişmiş Ayarlar")
use_vdd = st.sidebar.checkbox("Volatilite Dönüşüm Denklemi (VDD)", value=True)
use_ugk = st.sidebar.checkbox("Uyarlamalı Güven Katsayısı (UGK)", value=True)
ugk_conf = st.sidebar.slider("UGK güven katsayısı", 0.0, 1.0, 0.5)

# -------- Veri ve model --------
if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"**{symbol}** verisi indiriliyor ({start_date} – {end_date})…")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli kod ve tarih aralığı girin.")
    elif (end_date - start_date).days < 7 or data.shape[0] < 20:
        st.warning("En az 7 işlem günü (≈20 satır) gerekli. Tarih aralığını genişletin.")
    else:
        # -------- Gerçek zamanlı fiyat --------
        try:
            current_price = float(yf.Ticker(symbol).info["currentPrice"])
            st.info(f"Gerçek Zamanlı Fiyat: {current_price:.2f} TL")
        except:
            current_price = data.Close.iloc[-1]
            st.warning("Gerçek zamanlı fiyat alınamadı, son kapanış kullanılıyor.")

        # -------- Teknik göstergeler --------
        data["MA5"]  = data.Close.rolling(5).mean()
        data["MA10"] = data.Close.rolling(10).mean()
        data["RealTimePrice"] = current_price
        data["Target"] = data.Close.shift(-1)
        data.dropna(inplace=True)

        features = ["Close", "MA5", "MA10", "RealTimePrice"]
        X, y = data[features], data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.success(f"Model Ortalama Hata (MAE): ±{mae:.2f} TL")

        # -------- Ham tahmin (bugün & yarın) --------
        today_raw     = model.predict(X.iloc[[-1]])[0]
        tomorrow_raw  = model.predict(X.iloc[[-1]])[0]  # yarın için de son satırı baz al

        # -------- Piyasa verileri --------
        recent_diff   = float(data.Close.iloc[-1] - data.Close.iloc[-2])
        volatility    = float(data.Close.pct_change().rolling(5).std().iloc[-1] * 100)
        vol_factor    = min(max(volatility / 5, -1), 1)  # -1 … 1

        # -------- VDD düzeltmesi --------
        if use_vdd:
            vdd_adj = recent_diff * np.log1p(volatility / 100)
            today_raw    += vdd_adj
            tomorrow_raw += vdd_adj

        # -------- UGK düzeltmesi --------
        if use_ugk:
            ugk_adj = mae * (volatility / 100) * ugk_conf
            today_raw    += ugk_adj
            tomorrow_raw += ugk_adj

        # -------- Devre sınırı (%10) --------
        up_lim, low_lim = current_price * 1.10, current_price * 0.90
        today_pred    = max(min(today_raw,    up_lim), low_lim)
        tomorrow_pred = max(min(tomorrow_raw, up_lim), low_lim)

        # -------- Sonuç --------
        st.subheader("📈 Tahmin Sonuçları")
        st.write(f"**Bugün (kapanış) →** {today_pred:.2f} TL")
        st.write(f"**Yarın (kapanış) →** {tomorrow_pred:.2f} TL")

        with st.expander("Denklem Ayrıntıları"):
            st.markdown(f"""
- **Volatilite Dönüşüm Denklemi (VDD)**  
  `vdd_adj = ΔFiyat × log(1 + Volatilite%)`

- **Uyarlamalı Güven Katsayısı (UGK)**  
  `ugk_adj = MAE × (Volatilite% / 100) × GüvenKatsayısı`  
  <sub>GüvenKatsayısı = {ugk_conf}</sub>

- **Volatilite%**: Son 5 günlük getirilerin st. sapması × 100  
- **ΔFiyat**: Son 2 kapanış fiyatı farkı  
- **Devre Sınırı**: ±%10 (BIST)
""")
