# Laporan Proyek Machine Learning
### Nama : Khilda Fitria Nurultsani
### Nim : 211351070
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya angkat kali ini adalah perhitungan lemak tubuh yang diambil dari 14 parameter yang telah ditentukan, menurut peneliti Peningkatan lemak dalam tubuh manusia dapat berpengaruh dalam perubahan bentuk tubuh  manusia. Maka dari itu, saya selaku pembuat mencoba membuat pengukur kadar lemak sebagai tindakan agar anda dapat mengetahui jumlah lemak yang ada dalam tubuh.

## Business Understanding
Proyek ini memudahkan kita untuk memonitor lemak tubuh, obesitas dan untuk rencana pengaturan diet dalam program pelayanan kesehatan menggunakan algoritma Regresi Linear. 

### Problem Statements
- Kelebihan lemak dapat menimbulkan obesitas yang merupakan faktor resiko dalam penyakit kardiovaskuler karena dapat menyebabkan hipertensi dan timbulnya diabetes.
- Diet dapat mempengaruhi komposisi tubuh dalam jangka waktu singkat, seperti pada saat kekurangan air dan kelaparan ataupun dalam jangka waktu lama, seperti pada chronic overeating yang dapat meningkatkan simpanan lemak tubuh.

### Goals
- Untuk mengetahui kadar lemak dalam tubuh, sehingga bisa membantu memantau kondisi kesehatan.

    ### Solution statements
    - Dibuatkannya pengukur kadar lemak tubuh agar kita dapat mengkontrol makanan
    - 

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, dimana isi dari Body Fat Prediction Dataset ini yaitu hasil pengukuran lingkar tubuh 252 pria dan diteliti berdasarkan 14 atribut diatas. Dataset ini digunakan untuk menggambarkan teknik regresi. Pengukuran lemak tubuh yang akurat tidaklah mudah dan diinginkan adanya metode yang mudah untuk memperkirakan lemak tubuh yang tidak merepotkan atau mahal.

[Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset/data). 

### Variabel-variabel pada Body Fat Prediction Dataset adalah sebagai berikut:
- Density : merupakan kepadatan tubuh dalam penimbangan dibawah air.
- Age : merupakan umur dalam satuan tahun.
- Weight : merupakan berat badan dalam satuan pon.
- Height : merupakan tinggi badan dalam satuan inchi.
- Neck : merupakan ukuran lingkar leher dalam satuan cm.
- Chest : merupakan ukuran ukuran lingkar dada dalam satuan cm.
- Abdomen : merupakan ukuran lingkar perut dalam satuan cm.
- Hip : merupakan ukuran lingkar panggul dalam satuan cm.
- Thigh : merupakan ukuran lingkar paha dalam satuan cm.
- Knee : merupakan ukuran lingkar lutut dalam satuan cm.
- Ankle : merupakan ukuran lingkar pergelangan kaki dalam satuan cm.
- Biceps : merupakan ukuran lingkar tangan atas dalam satuan cm.
- Forearm : merupakan ukuran lingkar tangan bawah dalam satuan cm.
- Wrist : merupakan ukuran lingkar pergelangan tangan dalam satuan cm.

## Data Preparation
Data berdasarkan kaggle

Pertama import dulu library yang di butuh dengan memasukan perintah :
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

Kemudian agar dataset di dalam kaggle langsung bisa terhubung ke kaggle maka harus membuat token terlebih dahulu di akun kaggle dengan memasukan perintah : 
```bash
from google.colab import files
files.upload()
```
Setelah itu lalu masukan file token.

Berikutnya yaitu membuat direktori dengan memasukan perintah :
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Setelah itu kita panggil url dataset yang ada di website kaggle untuk didownload langsung ke google colab.
```bash
!kaggle datasets download -d fedesoriano/body-fat-prediction-dataset
```

Jika berhasil, selanjutnya kita ekstrak dataset yang sudah didownload dengan perintah :
```bash
!mkdir body-fat-prediction-dataset
!unzip body-fat-prediction-dataset.zip -d body-fat-prediction-dataset
!ls body-fat-prediction-dataset
```

Jika berhasil diekstrak, maka kita langsung dapat membuka dataset tersebut dengan perintah :
```bash
df = pd.read_csv('/content/body-fat-prediction-dataset/bodyfat.csv')
```

Lalu kita dapat melakukan beberapa exploratory daya analysis sederhana, mulai dari menampilkan isi
dari dataset bodyfat.csv dengan memasukan perintah :
```bash
df.head()
```

Jika ingin menampilkan jumlah data dan kolom yang ada di dataset, masukan perintah :
```bash
df.shape
```

Lalu kita cek tipe data dari masing-masing atribut/fitur dari dataset dari bodyfat.csv , masukan perintah :
```bash
df.info()
```

Untuk menampilkan detail informasi dari dataset, masukan perintah :
```bash
df.describe()
```

Jika ingin mengecek heatmap dari data kita ada yang kosong atau tidak, masukan perintah :
```bash
sns.heatmap(df.isnull())
```
```bash
out :
```
<img width="338" alt="Screenshot 2023-10-26 230013" src="https://github.com/khildafitria/machinelearning/assets/149028314/60ba8229-27ff-4895-a4f2-7f23b5730a3a">

## Visualisasi Data
Menggunakan heatmap untuk melihat sebaran data pada dataset bodyfat.csv , masukan perintah :
```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
```bash
out :
```
<img width="430" alt="image" src="https://github.com/khildafitria/machinelearning/assets/149028314/9724d627-cf6d-4794-8826-93098ef3690f">

## Modeling
Untuk melakukan modeling saya memakai algoritma regresi linear, dimana kita harus memisahkan mana saja atribut yang akan dijadikan sebagai fitur(x) dan atribut mana yang dijadikan label(y).
```bash
features = ['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm','Wrist']
x = df[features]
y = df['BodyFat']
x.shape, y.shape
```
Pada perintah tersebut kita gunakan Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, dan Wrist sebagai fitur inputan(x). Sedangkan BodyFat dijadikan sebagai label(y), karena BodyFat merupakan nilai yang akan diestimasi.


Berikutnya lakukan split data, yaitu memisahkan data training dan data testing dengan memasukan perintah :
```bash
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```

Lalu masukan data training dan testing ke dalam model regresi linier dengan perintah :
```bash
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```

Untuk mengecek akurasinya masukan perintah :
```bash
score = lr.score(X_test, y_test)
print('akurasi model regresi linear =', score)
```
```bash
out : akurasi model regresi linear = 0.9909700768437055
```
Didapatkan nilai akurasi 99% , hal ini dipengaruhi oleh jumlah parameter yang digunakan. Jika parameternya dikurangi maka tingkat akurasinya terpengaruh.


Selanjutnya mencoba menggunakan model estimasi menggunakan regresi linier dengan memasukan perintah :
```bash
input_data = np.array([[1.0708 , 23 ,	154.25 ,	67.75 ,	36.2 ,	93.1 ,	85.2 ,	94.5 ,	59.0	, 37.3 ,	21.9 ,	32.0 ,	27.4 ,	17.1]])
prediction = lr.predict(input_data)
print('Perkiraan Lemak Tubuh Dalam Persen :', prediction)
```
```bash
out : Perkiraan Lemak Tubuh Dalam Persen : [11.87458955]
```

Berdasarkan data yang telah diteliti, maka kita dapat mengetahui kadar lemak yang ada dalam tubuh kita.

## Evaluation
Metrik evaluasi yang digunakan yaitu precision dengan memasukan perintah :
```bash
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)

print(f"precision = {score}")
```
- Saya memilih kasus klasifikasi dan menggunakan metrik **precision**. Karena dalam mendeteksi lemak tubuh, kesalahan dalam mendeteksi lemak yang sebenarnya tidak ada dapat menyebabkan kecemasan yang tidak perlu atau biaya tambahan untuk tes lebih lanjut. Presisi membantu dalam mengukur sejauh mana model ini dapat menghindari kesalahan.
- 

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
[Perhitungan Lemak Tubuh](https://machinelearning-khilda.streamlit.app/). 

![image](https://github.com/khildafitria/machinelearning/assets/149028314/ce9faa42-8b9e-4fda-aff6-97d2734cd341)

