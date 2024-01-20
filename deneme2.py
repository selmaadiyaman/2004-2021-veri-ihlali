import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Veri çerçevesini yükleme
dosya_yolu = 'C:\\Users\\lenovo\\OneDrive\\Masaüstü\\DataBreaches(2004-2021).csv'
veri = pd.read_csv(dosya_yolu)

veri['Year'] = pd.to_datetime(veri['Year'], errors='coerce', format='%Y').dt.year.fillna(0).astype(int)
veri['Records'] = veri['Records'].fillna(0).astype(float)
print(veri.dtypes)
X = veri[['Year', 'Records']]  # Özellikler (Bağımsız değişkenler)
y = veri['Method']  # Hedef değişken (Bağımlı değişken)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lojistik regresyon modelini oluşturma ve eğitme
model = LogisticRegression(max_iter=1000)  # 1000 iterasyon örneği
model.fit(X_train, y_train)

# Test verisi üzerinde modelin performansını değerlendirme
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')  # 'micro', 'macro', 'weighted', veya None olarak değiştirilebilir

print(f'Hassasiyet (Precision): {precision:.2f}')
print(f'Duyarlılık (Recall): {recall:.2f}')
print(f'F1 Skoru: {f1:.2f}')
"""Modelimizde lojistik regresyon sınıflandırmasını kullandık. Modelimizin performans, hassasiyet (precision), duyarlılık (recall) ve f1 skoru yaklaşık olarak 0.52 çıktı. 
Bu değerler, modelin veriler üzerindeki performansının orta seviyede olduğunu gösterir."""

