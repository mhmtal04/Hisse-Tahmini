import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Gerekli kütüphaneleri kontrol et ve yoksa yükle
try:
    import yfinance as yf
except ImportError:
    install("yfinance")
    import yfinance as yf

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    install("scikit-learn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

# Kullanıcıdan hisse kodu al (örneğin: THYAO, ASELS, SISE gibi)
symbol = input("Hisse kodunu gir (örnek: THYAO): ").upper() + ".IS"

# Veriyi indir
print(f"{symbol} verisi indiriliyor...")
data = yf.download(symbol, period="6mo")

# Özellikleri hazırla
data["Return"] = data["Close"].pct_change()
data["Target"] = (data["Return"].shift(-1) > 0).astype(int)
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()
data = data.dropna()

# Modeli eğit
features = ["Close", "MA5", "MA10"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Başarı oranını göster
preds = model.predict(X_test)
print(f"Model Doğruluk Oranı: {accuracy_score(y_test, preds)*100:.2f}%")

# Yarın ne olur tahmini
latest_data = X.tail(1)
prediction = model.predict(latest_data)

print("\nTahmin Sonucu:")
if prediction[0] == 1:
    print("Hisse yarın + açacak gibi görünüyor.")
else:
    print("Hisse yarın - açacak gibi görünüyor.")