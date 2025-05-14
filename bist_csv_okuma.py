import pandas as pd

# CSV dosyasının yolunu belirtin (örneğin: 'BIST_veri.csv')
dosya_yolu = "bist_ornek.csv"  # Buraya kendi dosya adınızı yazın

# CSV'yi oku (Türkçe karakter içeren dosyalar için uygun encoding ve ayraç)
try:
    df = pd.read_csv(dosya_yolu, encoding="ISO-8859-9", sep=";")
    print("Veri başarıyla yüklendi.")
except Exception as e:
    print("Veri yüklenemedi:", e)
    exit()

# İlk 5 satırı göster
print("\nVerinin ilk 5 satırı:")
print(df.head())

# Örnek: ALTINS1 verisini filtrele
if "MENKUL KIYMET" in df.columns:
    altin_df = df[df["MENKUL KIYMET"] == "ALTINS1"]
    print("\nALTINS1 verileri:")
    print(altin_df)
else:
    print("\n'MENKUL KIYMET' kolonu bulunamadı. Lütfen doğru dosyayı kullandığınızdan emin olun.")