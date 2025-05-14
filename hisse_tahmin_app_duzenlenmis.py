import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

st.title("Hisse Tahmin Uygulaması")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

if symbol:
    symbol += ".IS"
    st.write(f"{symbol} verisi indiriliyor...")
    data = yf.download(symbol, period="6mo")

    if data.empty:
        st.warning("Veri indirilemedi. Hisse kodunu kontrol edin.")
    else:
        data["Return"] = data["Close"].pct_change()
        data["Target"] = (data["Return"].shift(-1) > 0).astype(int)
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data = data.dropna()

        if data.shape[0] < 20:
            st.warning("Yeterli veri yok. Daha uzun zaman dilimi seçin.")
        else:
            features = ["Close", "MA5", "MA10"]
            X = data[features]
            y = data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds) * 100
            st.success(f"Model Doğruluk Oranı: {accuracy:.2f}%")

            latest_data = X.tail(1)
            prediction = model.predict(latest_data)

            st.subheader("Tahmin:")
            if prediction[0] == 1:
                st.write("Hisse yarın **+ açacak** gibi görünüyor.")
            else:
                st.write("Hisse yarın **- açacak** gibi görünüyor.") 
