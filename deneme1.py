import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# CSV dosyasının yolunu belirtin
dosya_yolu = 'DataBreaches(2004-2021).csv'  # Gerçek dosya adını ve yolunu kullanın

# Veriyi CSV'den yükle
dosya_yolu = 'C:\\Users\\lenovo\\OneDrive\\Masaüstü\\DataBreaches(2004-2021).csv'
veri = pd.read_csv(dosya_yolu)

# Veri çerçevesini görüntüleyin
print(veri)

# Eksik değerlerin kontrolü
print(veri.isnull().sum())

# Anlamsız verilerin düzeltilmesi veya kaldırılması
veri['Year'] = pd.to_datetime(veri['Year'], errors='coerce', format='%Y')

# Veri biçimlendirme
veri['Year'] = pd.to_datetime(veri['Year'], errors='coerce', format='%Y').dt.year.fillna(0).astype(int)
veri['Records'] = veri['Records'].fillna(0).astype(float)
veri['Entity'] = veri['Entity'].fillna('Unknown')
veri['Organization type'] = veri['Organization type'].fillna('Unknown')
veri['Method'] = veri['Method'].fillna('Unknown')

print(veri.dtypes)

# Örnek veri çerçevesini kendisiyle birleştirelim (örneğin, ilk 10 satırı tekrar ekleyelim)
veri_birlestirilmis = pd.concat([veri, veri.head(10)], ignore_index=True)

print(veri_birlestirilmis)

# 'Year' sütununu indeks olarak ayarla
veri.set_index('Year', inplace=True)

# 2010 ile 2015 yılları arasındaki verileri seç (filtreleme)
veri_2010_2015 = veri[(veri.index >= 2010) & (veri.index <= 2015)]
print(veri_2010_2015)

# Hangi sütunların tekrar eden verilere sahip olduğunu belirleme
duplicated_rows = veri[veri.duplicated()]

print(duplicated_rows)

# Veriyi yıllara göre sırala
veri_sirali = veri_2010_2015.sort_index()

print(veri_sirali)

# 2010-2015 yılları arasında içeriden yapılan ihlalleri filtreleme
inside_job = veri_sirali[veri_sirali['Method'].str.contains('inside job', case=False, na=False)]


# Belirtilen yıllara ait içeriden yapılan ihlallerin sayısı
inside_job_count = inside_job.shape[0]

# Belirtilen yıllara ait toplam ihlal sayısı
total_records_count = veri_sirali.shape[0]

# Yüzdeyi hesaplama
if total_records_count != 0:
    percentage_inside_job = (inside_job_count / total_records_count) * 100
    print(f"2010-2015 yılları arasındaki veri ihlallerinin %{percentage_inside_job:.2f}'ü 'şirket içi sızdırılan veriler' olarak belirtilmiştir.")
else:
    print("Belirtilen yıl aralığında kayıt bulunmamaktadır.")

veri_sirali.reset_index(inplace=True)
correlation = veri_sirali['Year'].corr(veri_sirali['Records'])
print(f"Year ve Records arasındaki korelasyon: {correlation:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

# Veri çerçevesini yeniden yükleme ve 'Year' sütununu datetime olarak değiştirme
veri = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Masaüstü\\DataBreaches(2004-2021).csv')
veri['Year'] = pd.to_datetime(veri['Year'], errors='coerce', format='%Y').dt.year.fillna(0).astype(int)

# 2010 ile 2015 yılları arasındaki verileri seçme
veri_2010_2015 = veri.loc[(veri['Year'] >= 2010) & (veri['Year'] <= 2015)]

# Veri çerçevesini yıllara göre sıralama
veri_sirali = veri_2010_2015.sort_values(by='Year')

# Seaborn ile zaman çizelgesi oluşturma
plt.figure(figsize=(10, 6))
sns.lineplot(data=veri_sirali, x='Year', y='Records', marker='o')
plt.title('Yıllara Göre Şirket İçi Sızdırılan Veri İhlali Kayıtları')
plt.xlabel('Yıl')
plt.ylabel('Kayıt Sayısı')
plt.grid(True)
plt.show()


X = veri[['Year', 'Organization type']]  # Özellikler (Bağımsız değişkenler)
y = veri['Method']                      # Hedef değişken (Bağımlı değişken)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lojistik regresyon modelini oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde modelin performansını değerlendirme
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Hassasiyet (Precision): {precision:.2f}')
print(f'Duyarlılık (Recall): {recall:.2f}')
print(f'F1 Skoru: {f1:.2f}')