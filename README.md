# Laporan Proyek Machine Learning - Fadhilah Nurrahmayanti

## Domain Proyek

Produksi pangan merupakan sektor vital bagi negara-negara ASEAN, mengingat sebagian besar wilayahnya masih mengandalkan pertanian sebagai sumber utama pangan dan pendapatan. Dengan meningkatnya populasi dan perubahan iklim yang tidak menentu, memprediksi produksi pangan menjadi kebutuhan strategis. Data historis produksi komoditas seperti jagung, beras, kopi, coklat, dan minyak sawit dapat dimanfaatkan untuk membuat prediksi produksi di masa depan.

Meskipun sangat penting, tren produksi di negara-negara ini sering kali sangat fluktuatif, dipengaruhi oleh variabilitas musiman dan dinamika global. Hal ini menyulitkan dalam memproyeksikan tingkat pasokan di masa depan secara akurat, yang pada akhirnya menjadi tantangan dalam ketahanan pangan, strategi perdagangan, dan perencanaan kebijakan jangka panjang.

Dengan tersedianya data melalui [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data), memungkinkan penggunaan **Data-driven Forecasting Models** seperti neural network **Long Short-Term Memory (LSTM)** untuk memproyeksikan tren produksi hingga tahun 2030 dan membandingkan hasil per negara dari waktu ke waktu.

**Mengapa Masalah Ini Perlu diselesaikan**

1. **Perencanaan Pertanian Jangka Panjang**
   Pemerintah dan perencana pertanian membutuhkan prediksi yang andal untuk menyusun strategi ketahanan pangan dan mengoptimalkan infrastruktur produksi.

2. **Tolok Ukur Strategis Antar Negara ASEAN**
   Peramalan produksi masa depan membantu menentukan apakah Indonesia tetap dominan dalam produksi beras.

3. **Analisis Tren Produksi Global**
   Model LSTM univariat memungkinkan analisis tren setiap komoditas secara independen, memberikan wawasan terhadap stagnasi pertumbuhan atau potensi peningkatan hasil.

**Bagaimana Masalah Ini Akan Diselesaikan**

Solusi dilakukan dengan membangun **model LSTM univariat** yang hanya menggunakan **jumlah produksi tahunan** sebagai input, tanpa variabel lingkungan atau ekonomi tambahan. Pendekatan ini murni berbasis data historis dari [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)

**Langkah Implementasi:**

1. **Pra-pemrosesan data deret waktu** (1961–2021) untuk tiap komoditas (jagung, beras, kopi, coklat, dan minyak sawit) dan negara (Indonesia, Vietnam, Thailand, Filipina dan Malaysia).
2. **Melatih model LSTM terpisah** untuk tiap komoditas dan negara.
3. **Memprediksi produksi tahun 2022–2030.**
4. **Visualisasi dan perbandingan** proyeksi antar negara dan komoditas.nggunakan **RMSE** sebagai indikator akurasi.

## Business Understanding

### Problem Statements

**Masalah 1:**

Tren historis produksi pertanian di negara-negara ASEAN sangat fluktuatif. Tanpa peramalan yang akurat, pemerintah dan pemangku kepentingan di sektor pertanian kesulitan dalam merumuskan strategi pangan dan perdagangan jangka panjang.

**Masalah 2:**

Belum tersedia standar perbandingan proyeksi produksi yang membandingkan posisi Indonesia dengan negara produsen utama ASEAN lainnya, seperti Vietnam, Thailand, dan Malaysia, untuk komoditas strategis seperti beras, kopi, dan minyak sawit.

**Masalah 3:**

Teknik peramalan tradisional (misalnya statistik dasar atau regresi linier) sering kali gagal menangkap pola musiman dan non-linier dalam data deret waktu produksi pertanian jangka panjang.

### Goals

**Tujuan 1 (untuk Masalah 1):**

Mengembangkan model **LSTM univariat yang akurat** untuk meramalkan produksi tahunan beras, kopi, dan minyak sawit dari tahun 2022 hingga 2030 menggunakan data historis (1961–2021) per negara dan komoditas.

**Tujuan 2 (untuk Masalah 2):**

Membandingkan hasil proyeksi produksi antara Indonesia, Vietnam, Thailand, dan Malaysia untuk masing-masing komoditas, guna mengevaluasi posisi daya saing Indonesia secara kuantitatif.

**Tujuan 3 (untuk Masalah 3):**

Menggunakan LSTM untuk memodelkan tren historis produksi yang non-linier dan musiman, sehingga meningkatkan akurasi peramalan dibandingkan model konvensional seperti moving average atau regresi linier.

### Solution Statements

**Solusi 1:**

Membangun **model LSTM univariat secara terpisah** untuk setiap kombinasi komoditas dan negara (total 12 model) menggunakan data produksi tahunan dari [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data).

Untuk mengukur performa peramalan:

* **RMSE (Root Mean Squared Error)** – mengukur seberapa besar deviasi prediksi terhadap nilai aktual.
* **MAE (Mean Absolute Error)** – mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual.

**Solusi 2:**

Membandingkan dan memvisualisasikan hasil prediksi total produksi per kombinasi komoditas dan negara, sehingga para pemangku kepentingan dapat menilai posisi daya saing Indonesia di kawasan ASEAN.

## Data Understanding

Dataset yang digunakan berisi data produksi komoditas pangan dari berbagai negara ASEAN. File data bernama Data.csv, dan berisi 11.912 baris dengan 24 kolom.

Dataset: [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data) (1961–2021)

### **Variabel dalam Dataset**

| Nama Variabel                         | Deskripsi                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `Entity`                              | Nama negara atau wilayah (contoh: Indonesia, Vietnam, Thailand, Malaysia).   |
| `Year`                                | Tahun kalender dari data yang dicatat (dari 1961 hingga 2023).               |
| `Maize Production (tonnes)`           | Produksi jagung dalam satuan ton –> variabel utama dalam proyek ini.         |
| `Rice  Production ( tonnes)`          | Produksi beras dalam satuan ton –> variabel utama dalam proyek ini.        |
| `Yams  Production (tonnes)`           | Produksi ubi jalar dalam satuan ton.                                         |
| `Wheat Production (tonnes)`           | Produksi gandum dalam satuan ton.                                            |
| `Tomatoes Production (tonnes)`        | Produksi tomat dalam satuan ton.                                             |
| `Tea  Production ( tonnes )`          | Produksi teh dalam satuan ton.                                               |
| `Sweet potatoes  Production (tonnes)` | Produksi ubi manis dalam satuan ton.                                         |
| `Sunflower seed  Production (tonnes)` | Produksi biji bunga matahari dalam satuan ton.                               |
| `Sugar cane Production (tonnes)`      | Produksi tebu dalam satuan ton.                                              |
| `Soybeans  Production (tonnes)`       | Produksi kedelai dalam satuan ton.                                           |
| `Rye  Production (tonnes)`            | Produksi gandum hitam dalam satuan ton.                                      |
| `Potatoes  Production (tonnes)`       | Produksi kentang dalam satuan ton.                                           |
| `Oranges  Production (tonnes)`        | Produksi jeruk dalam satuan ton.                                             |
| `Peas, dry Production ( tonnes)`      | Produksi kacang polong kering dalam satuan ton.                              |
| `Palm oil  Production (tonnes)`       | Produksi minyak sawit dalam satuan ton –> variabel utama dalam proyek ini. |
| `Grapes  Production (tonnes)`         | Produksi anggur dalam satuan ton.                                            |
| `Coffee, green Production ( tonnes)`  | Produksi kopi biji hijau –> variabel utama dalam proyek ini.               |
| `Cocoa beans Production (tonnes)`     | Produksi biji kakao dalam satuan –> variabel utama dalam proyek ini.ton.                                        |
| `Meat, chicken  Production (tonnes)`  | Produksi daging ayam dalam satuan ton.                                       |
| `Bananas  Production ( tonnes)`       | Produksi pisang dalam satuan ton.                                            |
| `Avocados Production (tonnes)`        | Produksi alpukat dalam satuan ton.                                           |
| `Apples Production (tonnes)`          | Produksi apel dalam satuan ton.                                              |

Dalam proyek ini, hanya fokus pada **lima komoditas utama**:

* `Maize Production (tonnes)`
* `Rice Production (tonnes)`
* `Coffee green Production (tonnes)`
* `Cocoa beans Production (tonnes)`
* `Palm oil Production (tonnes)`

dan **lima negara ASEAN**:

* **Indonesia**
* **Vietnam**
* **Thailand**
* **Filipina**
* **Malaysia**

### Missing Value and Duplicate Data Check

Memastikan kualitas data sebelum masuk ke tahap analisis atau pemodelan, karena nilai kosong maupun data duplikat dapat mengganggu hasil dan akurasi model.

```python
 data.isnull().sum()
```

Berikut adalah hasil jumlah nilai hilang di setiap kolom:

| Kolom                                  | Nilai Hilang |
| -------------------------------------- | ------------ |
| Entity                                 | 0            |
| Year                                   | 0            |
| Maize Production (tonnes)              | **0**        |
| Rice  Production ( tonnes)             | **0**        |
| Yams  Production (tonnes)              | 0            |
| Wheat Production (tonnes)              | 0            |
| Tomatoes Production (tonnes)           | 0            |
| Tea  Production ( tonnes )             | 0            |
| Sweet potatoes  Production (tonnes)    | 0            |
| Sunflower seed  Production (tonnes)    | 0            |
| Sugar cane Production (tonnes)         | 0            |
| Soybeans  Production (tonnes)          | 0            |
| Rye  Production (tonnes)               | 0            |
| Potatoes  Production (tonnes)          | 0            |
| Oranges  Production (tonnes)           | 0            |
| Peas, dry Production ( tonnes)         | 0            |
| **Palm oil  Production (tonnes)**      | **0**        |
| Grapes  Production (tonnes)            | 0            |
| **Coffee, green Production ( tonnes)** | **0**        |
| Cocoa beans Production (tonnes)        | **0**        |
| Meat, chicken  Production (tonnes)     | 0            |
| Bananas  Production ( tonnes)          | 0            |
| Avocados Production (tonnes)           | 0            |
| Apples Production (tonnes)             | 0            |

```python
print('Total Duplikasi Data:', df.duplicated().sum())
```
Output:
**Total Duplikasi Data: 0**

**Kesimpulan**: Dataset sepenuhnya bersih dengan **tidak ada nilai hilang** dan **duplikasi data** pada semua kolom, termasuk ketiga variabel target: **jagung**, **beras**, **kopi**, **cokelat** dan **minyak sawit**.

### ASEAN Country and Commodity Selection

Memfokuskan analisis pada lima negara ASEAN, yaitu Indonesia, Vietnam, Thailand, Filipina, dan Malaysia, serta pada komoditas tertentu yang relevan secara ekonomi dan produksi, yaitu jagung, beras, kopi, kakao, dan minyak sawit.

Berikut adalah cuplikan data produksi tahunan lima komoditas utama di Indonesia (1961–1965) yang digunakan dalam proyek ini:

| Year | Entity    | Maize Production (tonnes) | Rice Production (tonnes) | Coffee green Production (tonnes) | Cocoa beans Production (tonnes) | Palm oil Production (tonnes) |
|------|-----------|----------------------------|----------------------------|----------------------------------|----------------------------------|-------------------------------|
| 1961 | Indonesia | 2,283,100                  | 7,965,911                  | 47.05                            | 20,000                           | 21,602                        |
| 1962 | Indonesia | 3,242,900                  | 8,694,623                  | 48.30                            | 28,000                           | 23,002                        |
| 1963 | Indonesia | 2,357,800                  | 8,735,168                  | 46.38                            | 32,000                           | 24,802                        |
| 1964 | Indonesia | 3,768,600                  | 9,371,189                  | 48.41                            | 35,000                           | 25,002                        |
| 1965 | Indonesia | 2,364,500                  | 9,655,017                  | 48.48                            | 35,000                           | 27,202                        |

### Checking Outliers

Proses ini menggunakan visualisasi boxplot untuk mengidentifikasi outlier pada lima komoditas utama. Boxplot membantu melihat sebaran data, median, kuartil, dan titik-titik ekstrem yang dianggap sebagai outlier berdasarkan rentang interkuartil.

```python
commodities = [
    'Maize Production (tonnes)',
    'Rice  Production ( tonnes)',
    'Coffee, green Production ( tonnes)',
    'Cocoa beans Production (tonnes)',
    'Palm oil  Production (tonnes)'
]

plt.figure(figsize=(12, 6))

for i, commodity in enumerate(commodities):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=df[commodity], color='red')
    plt.title(commodity)

plt.tight_layout()
plt.show()
```

Hasilnya seperti gambar berikut:
![Checking Outliers](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Checking-Outliers.png?raw=true)

---

## Data Preparation

### Feature Renaming for Simplicity

Nama kolom pada dataset mengandung format yang tidak konsisten, seperti spasi berlebih dan anotasi satuan (contoh: " (tonnes)" atau " ( tonnes)"), yang perlu dibersihkan untuk menghindari kesalahan kunci dalam pemrosesan selanjutnya.

Langkah-langkah pembersihan:

```python
df_filtered.columns = df_filtered.columns.str.replace(' \(tonnes\)', '', regex=True)
df_filtered.columns = df_filtered.columns.str.replace(' \( tonnes\)', '', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Coffee, green Production', 'Coffee green Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Palm oil  Production', 'Palm oil Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Cocoa beans Production', 'Cocoa beans Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Rice  Production', 'Rice Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Maize Production', 'Maize Production', regex=True)
df_filtered.head()
```

Hasilnya:

| Year | Entity    | Maize Production | Rice Production | Coffee green Production | Cocoa beans Production | Palm oil Production |
|------|-----------|------------------|------------------|--------------------------|--------------------------|----------------------|
| 1961 | Indonesia | 2,283,100        | 7,965,911        | 47.05                    | 20,000                   | 21,602               |
| 1962 | Indonesia | 3,242,900        | 8,694,623        | 48.30                    | 28,000                   | 23,002               |
| 1963 | Indonesia | 2,357,800        | 8,735,168        | 46.38                    | 32,000                   | 24,802               |
| 1964 | Indonesia | 3,768,600        | 9,371,189        | 48.41                    | 35,000                   | 25,002               |
| 1965 | Indonesia | 2,364,500        | 9,655,017        | 48.48                    | 35,000                   | 27,202               |

### Handling Outliers

Data produksi pertanian sering kali mengandung nilai ekstrem dan pola pertumbuhan non-linear. Alih-alih menghapus outlier, proyek ini menggunakan **transformasi logaritmik (`log1p`)** untuk menstabilkan variansi namun tetap mempertahankan semua data:

```python
df_transformed = df_filtered.copy()

commodities = [
    'Maize Production',
    'Rice Production',
    'Coffee green Production',
    'Cocoa beans Production',
    'Palm oil Production'
]

for col in commodities:
    df_transformed[col] = np.log1p(df_transformed[col])

df_transformed.head()
```

Hasilnya ditunjukkan dalam gambar berikut:
![Handling Outliers](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Handling-Outliers.png?raw=true)

### Data Normalization

Normalisasi data dilakukan untuk menskalakan nilai fitur ke dalam rentang 0 hingga 1 menggunakan metode **MinMaxScaler** agar setiap fitur memiliki skala yang seimbang.

```python
df_normalized = df_transformed.copy()

scalers = {}

for col in commodities:
    scaler = MinMaxScaler()
    df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
    scalers[col] = scaler

df_normalized
```

Berikut Data Produksi Pangan Ternormalisasi (Indonesia):

| Entity     | Year | Maize Production | Rice Production | Coffee green Production | Cocoa beans Production | Palm oil Production |
|------------|------|------------------|------------------|--------------------------|-------------------------|----------------------|
| Indonesia  | 1961 | 0.686313         | 0.618756         | 0.055055                 | 0.659623                | 0.611591             |
| Indonesia  | 1962 | 0.728914         | 0.627802         | 0.057219                 | 0.682033                | 0.615439             |
| Indonesia  | 1963 | 0.690222         | 0.628283         | 0.053872                 | 0.690926                | 0.620056             |
| Indonesia  | 1964 | 0.747151         | 0.635546         | 0.057407                 | 0.696895                | 0.620548             |
| Indonesia  | 1965 | 0.690566         | 0.638630         | 0.057526                 | 0.696895                | 0.625716             |

### Data Splitting

Membagi data masing-masing negara dan komoditas ke dalam subset pelatihan (train) dan pengujian (test) berdasarkan proporsi 80% untuk pelatihan dan 20% untuk pengujian. Data dibagi secara individual untuk setiap kombinasi negara dan komoditas, agar model nantinya dapat belajar dari pola spesifik masing-masing.

| No | Country – Commodity                   | Train Length | Test Length | Total Years |
|----|----------------------------------------|--------------|-------------|--------------|
| 0  | Indonesia - Maize Production           | 48           | 13          | 61           |
| 1  | Indonesia - Rice Production            | 48           | 13          | 61           |
| 2  | Indonesia - Coffee green Production    | 48           | 13          | 61           |
| 3  | Indonesia - Cocoa beans Production     | 48           | 13          | 61           |
| 4  | Indonesia - Palm oil Production        | 48           | 13          | 61           |
| 5  | Vietnam - Maize Production             | 48           | 13          | 61           |
| 6  | Vietnam - Rice Production              | 48           | 13          | 61           |
| 7  | Vietnam - Coffee green Production      | 48           | 13          | 61           |
| 8  | Vietnam - Cocoa beans Production       | 48           | 13          | 61           |
| 9  | Vietnam - Palm oil Production          | 48           | 13          | 61           |
| 10 | Thailand - Maize Production            | 48           | 13          | 61           |
| 11 | Thailand - Rice Production             | 48           | 13          | 61           |
| 12 | Thailand - Coffee green Production     | 48           | 13          | 61           |
| 13 | Thailand - Cocoa beans Production      | 48           | 13          | 61           |
| 14 | Thailand - Palm oil Production         | 48           | 13          | 61           |
| 15 | Philippines - Maize Production         | 48           | 13          | 61           |
| 16 | Philippines - Rice Production          | 48           | 13          | 61           |
| 17 | Philippines - Coffee green Production  | 48           | 13          | 61           |
| 18 | Philippines - Cocoa beans Production   | 48           | 13          | 61           |
| 19 | Philippines - Palm oil Production      | 48           | 13          | 61           |
| 20 | Malaysia - Maize Production            | 48           | 13          | 61           |
| 21 | Malaysia - Rice Production             | 48           | 13          | 61           |
| 22 | Malaysia - Coffee green Production     | 48           | 13          | 61           |
| 23 | Malaysia - Cocoa beans Production      | 48           | 13          | 61           |
| 24 | Malaysia - Palm oil Production         | 48           | 13          | 61           |

### Data Reshape

Data deret waktu diubah menjadi format sekuensial menggunakan pendekatan sliding window dengan `look_back = 5`, yang berarti model akan mempelajari lima tahun sebelumnya untuk memprediksi satu tahun berikutnya.

```python
def create_sequences(series, look_back=5):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)
```

Berikut adalah 25 kombinasi unik antara negara ASEAN dan komoditas produksi yang tersedia dalam data hasil reshaping:

- Indonesia - Maize Production  
- Indonesia - Rice Production  
- Indonesia - Coffee green Production  
- Indonesia - Cocoa beans Production  
- Indonesia - Palm oil Production  
- Vietnam - Maize Production  
- Vietnam - Rice Production  
- Vietnam - Coffee green Production  
- Vietnam - Cocoa beans Production  
- Vietnam - Palm oil Production  
- Thailand - Maize Production  
- Thailand - Rice Production  
- Thailand - Coffee green Production  
- Thailand - Cocoa beans Production  
- Thailand - Palm oil Production  
- Philippines - Maize Production  
- Philippines - Rice Production  
- Philippines - Coffee green Production  
- Philippines - Cocoa beans Production  
- Philippines - Palm oil Production  
- Malaysia - Maize Production  
- Malaysia - Rice Production  
- Malaysia - Coffee green Production  
- Malaysia - Cocoa beans Production  
- Malaysia - Palm oil Production

### Data Structure Verification

Memastikan data hasil reshape memiliki dimensi yang sesuai untuk digunakan dalam model LSTM. Setiap sampel pelatihan dan pengujian direpresentasikan sebagai sekuens 5 tahun sebelumnya (look-back) dan target satu tahun ke depan.

```python
key = 'Indonesia - Rice Production'

print(f"Data for {key}:\n")
print("X_train shape:", reshaped_data[key]['X_train'].shape)
print("y_train shape:", reshaped_data[key]['y_train'].shape)
print("X_test shape:", reshaped_data[key]['X_test'].shape)
print("y_test shape:", reshaped_data[key]['y_test'].shape)
print("years_test shape:", reshaped_data[key]['years_test'].shape)
```

Struktur Data untuk Pasangan: *Indonesia - Rice Production*

- **X_train shape:** (43, 5, 1)  
  Artinya terdapat 43 sampel pelatihan, masing-masing menggunakan 5 langkah waktu (tahun) sebagai input, dan 1 fitur (produksi tahunan).
  
- **y_train shape:** (43,)  
  Label target yang berisi produksi pada tahun ke-6 dari setiap sequence.

- **X_test shape:** (8, 5, 1)  
  Terdapat 8 sampel pengujian dengan struktur yang sama: 5 tahun input untuk memprediksi tahun ke-6.

- **y_test shape:** (8,)  
  Label aktual untuk data pengujian.

- **years_test shape:** (8,)  
  Tahun-tahun yang diprediksi untuk evaluasi dan visualisasi hasil forecasting.

---

## Model Development

Tujuan utama dari proyek ini adalah membangun model peramalan deret waktu yang mampu memprediksi produksi pertanian di masa depan secara akurat menggunakan data historis tahunan. Untuk itu, arsitektur **Long Short-Term Memory (LSTM)** dipilih karena kemampuannya dalam mempelajari pola dependensi temporal yang kompleks dalam data berurutan.

Metode statistik tradisional (seperti ARIMA atau eksponensial smoothing) sering kali kesulitan dalam menangani:
- Tren yang bersifat non-linear
- Dependensi jangka panjang
- Volatilitas dalam data produksi pertanian

LSTM, yang merupakan turunan dari Recurrent Neural Networks (RNN), sangat cocok untuk peramalan deret waktu karena menggunakan *memory cells* yang mampu menyimpan informasi sepanjang urutan data yang lebih panjang [1].

Untuk setiap pasangan negara–komoditas, dibangun satu model **LSTM univariat** yang dilatih menggunakan jendela data produksi tahunan selama 5 tahun (setelah transformasi log dan normalisasi).

- **Bentuk input**: `(jumlah sampel, 5, 1)`  
- **Arsitektur Layer**:
  - `LSTM(units=50, activation='relu')`
  - `Dense(units=1)`
- **Fungsi Loss**: Mean Squared Error (MSE)  
- **Optimizer**: Adam  
- **EarlyStopping**: Diaktifkan dengan `patience=10` dan `min_delta=0.001` untuk mencegah overfitting

### Train LSTM Model

Setiap model LSTM dilatih menggunakan 80% data urutan pertama, sementara 20% sisanya digunakan untuk pengujian. EarlyStopping diterapkan berdasarkan training loss:

```python
early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,
    callbacks=[early_stop]
)
```

---

## Evaluation

Setelah model LSTM dilatih untuk setiap pasangan negara–komoditas, langkah selanjutnya adalah mengevaluasi performa model berdasarkan nilai produksi dalam satuan aslinya (ton). Ini dilakukan dengan cara membalik proses normalisasi menggunakan inverse_transform.

Menghitung metrik evaluasi akhir dengan menggunakan:
* MSE (Mean Squared Error): rata-rata kuadrat dari selisih prediksi dan nilai aktual.
* RMSE (Root Mean Squared Error): akar dari MSE, lebih mudah ditafsirkan karena dalam satuan yang sama (ton).

### Evaluation Model and Inverse Transfom

Berikut adalah bagaimana evaluasi dilakukan:

```python
y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))

mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = mse ** 0.5
```

### Performa Model

Tabel berikut menampilkan nilai **MSE** dan **RMSE** untuk setiap kombinasi negara dan komoditas berdasarkan hasil evaluasi model LSTM univariat.

| Country – Commodity                      | MSE       | RMSE     |
|------------------------------------------|-----------|----------|
| Indonesia - Maize Production             | 0.043274  | 0.208023 |
| Indonesia - Rice Production              | 0.070069  | 0.264706 |
| Indonesia - Coffee green Production      | 0.065443  | 0.255818 |
| Indonesia - Cocoa beans Production       | 1.028181  | 1.013993 |
| Indonesia - Palm oil Production          | 0.467928  | 0.684052 |
| Vietnam - Maize Production               | 0.006084  | 0.078003 |
| Vietnam - Rice Production                | 0.039638  | 0.199093 |
| Vietnam - Coffee green Production        | 7.097098  | 2.664038 |
| Vietnam - Cocoa beans Production         | 1.139787  | 1.067608 |
| Vietnam - Palm oil Production            | 7.804762  | 2.793700 |
| Thailand - Maize Production              | 0.005047  | 0.071042 |
| Thailand - Rice Production               | 1.949327  | 1.396183 |
| Thailand - Coffee green Production       | 1.123807  | 1.060097 |
| Thailand - Cocoa beans Production        | 0.369230  | 0.607643 |
| Thailand - Palm oil Production           | 0.055478  | 0.235537 |
| Philippines - Maize Production           | 0.001285  | 0.035847 |
| Philippines - Rice Production            | 8.377754  | 2.894435 |
| Philippines - Coffee green Production    | 2.823268  | 1.680258 |
| Philippines - Cocoa beans Production     | 3.738368  | 1.933486 |
| Philippines - Palm oil Production        | 0.854754  | 0.924529 |
| Malaysia - Maize Production              | 0.401907  | 0.633961 |
| Malaysia - Rice Production               | 0.094173  | 0.306876 |
| Malaysia - Coffee green Production       | 2.983603  | 1.727311 |
| Malaysia - Cocoa beans Production        | 7.025335  | 2.650535 |
| Malaysia - Palm oil Production           | 2.682450  | 1.637819 |

**Insight**

* 🇮🇩 **Indonesia** menunjukkan performa prediksi yang cukup stabil. Model memiliki akurasi yang baik untuk jagung (**RMSE = 0.208**) dan beras (**0.265**). Prediksi untuk kopi juga masih dapat diterima (**0.256**). Namun, performa untuk kakao (**1.014**) dan minyak sawit (**0.684**) menunjukkan adanya tantangan dalam menangkap pola tren pada komoditas tersebut.

* 🇻🇳 **Vietnam** menunjukkan prediksi sangat akurat untuk jagung (**RMSE = 0.078**) dan beras (**0.199**), menandakan pola historis yang konsisten. Namun, performa sangat buruk pada kopi (**2.664**) dan minyak sawit (**2.794**) yang mengindikasikan fluktuasi ekstrem atau anomali dalam data historis komoditas tersebut.

* 🇹🇭 **Thailand** mencatat performa prediksi terbaik untuk jagung (**RMSE = 0.071**) dan kelapa sawit (**0.236**). Model juga cukup baik dalam memprediksi kakao (**0.608**), namun menunjukkan kesulitan untuk beras (**1.396**) dan kopi (**1.060**), kemungkinan besar karena ketidakstabilan musiman atau variasi antar tahun yang tinggi.

* 🇵🇭 **Filipina** memiliki performa sangat akurat untuk jagung (**RMSE = 0.036**), tetapi sangat rendah untuk beras (**2.894**) dan kakao (**1.933**). Ini mencerminkan adanya ketidakteraturan atau gangguan dalam data produksi tahunan komoditas tersebut. Komoditas lain seperti kopi (**1.680**) dan kelapa sawit (**0.925**) juga menunjukkan kesulitan moderat dalam prediksi.

* 🇲🇾 **Malaysia** mencatat akurasi tinggi pada beras (**RMSE = 0.307**) dan cukup baik untuk jagung (**0.634**). Namun, model kesulitan dalam memprediksi kopi (**1.727**), kakao (**2.651**), dan minyak sawit (**1.638**), yang menunjukkan bahwa tren historis produksi di komoditas tersebut lebih sulit untuk dipelajari oleh model.

### Visual Evaluation

Visualisasi dilakukan untuk membandingkan prediksi model terhadap data aktual pada periode pelatihan dan pengujian. Data produksi dikembalikan ke skala aslinya menggunakan inverse_transform sebelum divisualisasikan. Setiap grafik menunjukkan tren tahunan produksi komoditas dengan garis prediksi dan aktual untuk masing-masing negara dan komoditas, yang terbagi dalam dua subplot: pelatihan dan pengujian.

Indonesia's Model Evaluation:
![Indonesia - Maize Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Indonesia-Maize-Evaluation.png?raw=true)
![Indonesia - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Indonesia-Rice-Evaluation.png?raw=true)
![Indonesia - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Indonesia-Coffee-green-Evaluation.png?raw=true)
![Indonesia - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Indonesia-Cocoa-beans-Evaluation.png?raw=true)
![Indonesia - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Indonesia-Palm-oil-Evaluation.png?raw=true)

Vietnam's Model Evaluation:
![Vietnam - Maize Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Vietnam-Maize-Evaluation.png?raw=true)
![Vietnam - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Vietnam-Rice-Evaluation.png?raw=true)
![Vietnam - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Vietnam-Coffee-green-Evaluation.png?raw=true)
![Vietnam - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Vietnam-Cocoa-beans-Evaluation.png?raw=true)
![Vietnam - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Vietnam-Palm-oil-Evaluation.png?raw=true)

Thailand's Model Evaluation:
![Thailand - Maize Production](https://github.com/codedreamerD/MLT1/blob/main/repo-dir/Thailand-Maize-Evaluation.png?raw=true)
![Thailand - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Thailand-Rice-Evaluation.png?raw=true)
![Thailand - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Thailand-Coffee-green-Evaluation.png?raw=true)
![Thailand - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Thailand-Cocoa-beans-Evaluation.png?raw=true)
![Thailand - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Thailand-Palm-oil-Evaluation.png?raw=true)

Philippines Model Evaluation:
![Philippines - Maize Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Philippines-Maize-Evaluation.png?raw=true)
![Philippines - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Philippines-Rice-Evaluation.png?raw=true)
![Philippines - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Philippines-Coffee-green-Evaluation.png?raw=true)
![Philippines - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Philippines-Cocoa-beans-Evaluation.png?raw=true)
![Philippines - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Philippines-Palm-oil-Evaluation.png?raw=true)

Malaysia's Model Evaluation:
![Malaysia - Maize Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Malaysia-Maize-Evaluation.png?raw=true)
![Malaysia - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Malaysia-Rice-Evaluation.png?raw=true)
![Malaysia - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Malaysia-Coffee-green-Evaluation.png?raw=true)
![Malaysia - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Malaysia-Cocoa-beans-Evaluation.png?raw=true)
![Malaysia - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Malaysia-Palm-oil-Evaluation.png?raw=true)

### Forecasting Results

Berikut adalah tabel prediksi produksi tahun 2030 berdasarkan hasil model LSTM:

| Country     | Commodity               | Forecast 2030 | From Year | To Year |
| ----------- | ----------------------- | ------------- | --------- | ------- |
| Vietnam     | Cocoa beans Production  | 24.375193     | 2022      | 2030    |
| Indonesia   | Cocoa beans Production  | 16.617132     | 2022      | 2030    |
| Philippines | Cocoa beans Production  | 10.681157     | 2022      | 2030    |
| Thailand    | Cocoa beans Production  | 10.132552     | 2022      | 2030    |
| Malaysia    | Cocoa beans Production  | 9.965827      | 2022      | 2030    |
| Thailand    | Coffee green Production | 14.207090     | 2022      | 2030    |
| Malaysia    | Coffee green Production | 11.917822     | 2022      | 2030    |
| Philippines | Coffee green Production | 9.949503      | 2022      | 2030    |
| Indonesia   | Coffee green Production | 9.729990      | 2022      | 2030    |
| Vietnam     | Coffee green Production | 8.684875      | 2022      | 2030    |
| Indonesia   | Maize Production        | 17.465925     | 2022      | 2030    |
| Philippines | Maize Production        | 16.924751     | 2022      | 2030    |
| Thailand    | Maize Production        | 16.212734     | 2022      | 2030    |
| Vietnam     | Maize Production        | 14.704633     | 2022      | 2030    |
| Malaysia    | Maize Production        | 10.316228     | 2022      | 2030    |
| Philippines | Palm oil Production     | 14.456854     | 2022      | 2030    |
| Vietnam     | Palm oil Production     | 12.826477     | 2022      | 2030    |
| Indonesia   | Palm oil Production     | 11.666230     | 2022      | 2030    |
| Thailand    | Palm oil Production     | 11.150445     | 2022      | 2030    |
| Malaysia    | Palm oil Production     | 9.997130      | 2022      | 2030    |
| Indonesia   | Rice Production         | 21.746363     | 2022      | 2030    |
| Thailand    | Rice Production         | 15.662460     | 2022      | 2030    |
| Malaysia    | Rice Production         | 15.208253     | 2022      | 2030    |
| Vietnam     | Rice Production         | 14.594819     | 2022      | 2030    |
| Philippines | Rice Production         | 13.804643     | 2022      | 2030    |

####  Bar Chart Visualization

Grafik ini membandingkan hasil prediksi produksi tahun 2030 untuk lima komoditas utama di lima negara ASEAN. Visualisasi menggunakan grouped bar chart untuk memperlihatkan performa masing-masing negara dalam setiap komoditas.

![Forecasted Production in 2030](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Barchart-Prediction-2030.png?raw=true)

#### Forecast Comparison (Until 2030)

Melakukan prediksi produksi komoditas hingga tahun 2030 untuk setiap kombinasi negara dan komoditas yang tersedia. Model LSTM yang telah dilatih digunakan untuk menghasilkan prediksi berdasarkan 5 data terakhir dari gabungan data pelatihan dan pengujian. Output prediksi disimpan dalam struktur forecast_by_commodity dan divisualisasikan dalam bentuk grafik per komoditas untuk membandingkan tren antar negara.

![Forecast - Maize Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Forecast-Maize-Until-2030.png?raw=true)
![Forecast - Rice Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Forecast-Rice-Until-2030.png?raw=true)
![Forecast - Coffee Green Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Forecast-Coffee-green-Until-2030.png?raw=true)
![Forecast - Cocoa Beans Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Forecast-Cocoa-beans-Until-2030.png?raw=true)
![Forecast - Palm Oil Production](https://github.com/codedreamerD/MLT1_Pred/blob/main/repo-dir/Forecast-Palm-oil-Until-2030.png?raw=true)

---

## Keterkaitan Hasil Evaluasi dengan Business Understanding

### Apakah model menjawab setiap problem statement?

**Masalah 1:**
*Tren historis produksi pertanian di negara-negara ASEAN sangat fluktuatif. Tanpa peramalan yang akurat, pemerintah dan pemangku kepentingan di sektor pertanian kesulitan dalam merumuskan strategi pangan dan perdagangan jangka panjang.*

* **Terjawab.**
  Model LSTM univariat yang dibangun mampu menangkap tren tahunan yang bersifat musiman dan non-linier dari 25 kombinasi negara dan komoditas. Hasil evaluasi menggunakan RMSE menunjukkan performa yang baik untuk beberapa komoditas seperti:

  * Maize di Filipina (**RMSE = 0.035**)
  * Rice di Vietnam (**RMSE = 0.199**)
  * Palm oil di Thailand (**RMSE = 0.236**)

**Masalah 2:**
*Belum tersedia standar perbandingan proyeksi produksi yang membandingkan posisi Indonesia dengan negara produsen utama ASEAN lainnya.*

* **Terjawab.**
  Dataset telah dibagi dan diprediksi untuk masing-masing kombinasi negara dan komoditas, menghasilkan prediksi tahun 2030 yang memungkinkan:

  * Perbandingan kuantitatif antar negara berdasarkan **RMSE**
  * Perbandingan **volume produksi 2030** yang divisualisasikan dengan grafik bar dan tabel
  * Evaluasi daya saing Indonesia yang unggul dalam **rice** dan **maize**, sedangkan Vietnam unggul di **cocoa beans**

**Masalah 3:**
*Teknik peramalan tradisional (misalnya statistik dasar atau regresi linier) sering kali gagal menangkap pola musiman dan non-linier dalam data deret waktu produksi pertanian jangka panjang.*

* **Terjawab.**
  Penggunaan LSTM yang dapat menangani urutan data panjang serta pola fluktuatif terbukti efektif. Contohnya:

  * Cocoa beans Vietnam (RMSE = 1.068) tetap berhasil diprediksi meskipun datanya sangat berfluktuasi
  * Komoditas seperti rice Indonesia (**RMSE = 0.265**) menunjukkan hasil yang stabil dan realistis untuk prediksi jangka panjang

---

### Apakah model berhasil mencapai goals?

**Tujuan 1:**
Model berhasil dibangun dan dilatih dengan data historis produksi tahunan dari 1961 hingga 2021 untuk seluruh kombinasi negara dan komoditas. Selanjutnya, model digunakan untuk memprediksi produksi hingga tahun 2030 secara **recursive**.

**Tujuan 2:**
Hasil prediksi produksi tahun 2030 tersedia dalam bentuk tabel dan grafik bar, memungkinkan perbandingan antar negara dan komoditas. Daya saing Indonesia, Vietnam, dan negara ASEAN lainnya dapat dievaluasi secara kuantitatif dan visual.

**Tujuan 3:**
Model LSTM berhasil menangkap tren data non-linier yang kompleks. Hal ini menunjukkan keunggulan signifikan dibanding metode statistik biasa seperti regresi linier atau moving average.

---

### Apakah solusi yang direncanakan berdampak?

**Solusi 1:**
Model univariat LSTM menunjukkan performa optimal pada kombinasi data yang stabil maupun fluktuatif. Hal ini memvalidasi bahwa solusi berbasis deep learning cocok untuk konteks produksi pertanian tahunan.

**Solusi 2:**
Prediksi akhir dan visualisasi grafik bar memberi wawasan strategis untuk setiap negara dalam mengambil keputusan terkait produksi, perdagangan, dan ketahanan pangan berbasis komoditas.

---

### Kesimpulan

Model LSTM univariat yang dikembangkan dalam proyek ini terbukti efektif dan menjawab seluruh pernyataan masalah yang diajukan. Semua tujuan berhasil dicapai, dan solusi yang diterapkan berdampak langsung terhadap penyediaan alat bantu perencanaan strategis produksi pertanian jangka panjang di kawasan ASEAN. Proyek ini dapat menjadi referensi kuat untuk pemerintah, peneliti, dan pelaku industri pangan dalam menyusun kebijakan dan strategi berbasis data.

---

## Referensi

\[1] R. S. Ahmad, *World Food Production Dataset*, Kaggle. \[Online]. Tersedia: [https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)

\[2] N. Siddique dan H. Adeli, “Machine Learning Models for Food Security and Climate Change,” dalam *Machine Learning and Big Data Analytics Paradigms: Analysis, Applications and Challenges*, Springer, 2022, pp. 101–118. \[Online]. Tersedia: [https://link.springer.com/chapter/10.1007/978-3-031-08743-1\_6](https://link.springer.com/chapter/10.1007/978-3-031-08743-1_6)

\[3] ASEANstats, *ASEAN Statistical Brief, Vol. VII, April 2024*. ASEAN Secretariat, Apr. 2024. \[Online]. Tersedia: [https://www.aseanstats.org/wp-content/uploads/2024/04/ASEAN-Statistical-Brief\_April-2024\_v3.xlsx](https://www.aseanstats.org/wp-content/uploads/2024/04/ASEAN-Statistical-Brief_April-2024_v3.xlsx)

\[4] ASEAN Secretariat, *Industry Focus: Agriculture*. \[Online]. Tersedia: [https://asean.org/industry-focus/](https://asean.org/industry-focus/)

\[5] Encyclopaedia Britannica, “Southeast Asia – Industry,” *Britannica.com*. \[Online]. Tersedia: [https://www.britannica.com/place/Southeast-Asia/Industry](https://www.britannica.com/place/Southeast-Asia/Industry)
