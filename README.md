# Laporan Proyek Machine Learning
### Nama : Khilda Fitria Nurultsani
### Nim : 211351070
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya angkat kali ini adalah perhitungan lemak tubuh yang diambil dari 14 parameter yang telah ditentukan, menurut peneliti Peningkatan lemak dalam tubuh manusia dapat berpengaruh dalam perubahan bentuk tubuh  manusia. Maka dari itu, saya selaku pembuat mencoba membuat pengukur kadar lemak sebagai tindakan agar anda dapat mengetahui jumlah lemak yang ada dalam tubuh.

## Business Understanding
Proyek ini memudahkan kita untuk memonitor lemak tubuh, obesitas dan untuk rencana pengaturan diet dalam program pelayanan kesehatan menggunakan algoritma Regresi Linear. 

### Problem Statements
Seseorang bisa saja terkena obesitas dari parameter sebagai berikut :
- Kepadatan
- Usia
- Berat Badan
- Tinggi Badan
- Ukuran Lingkar Leher
- Ukuran Lingkar Dada
- Ukuran Lingkar Perut
- Ukuran Lingkar Panggul
- Ukuran Lingkar Paha
- Ukuran Lingkar Lutut
- Ukuran Lingkar Pergelangan Kaki
- Ukuran Lingkar Lengan Atas
- Ukuran Lingkar Lengan Bawah
- Ukuran Lingkar Pergelangan Tangan
  
Penelitian diatas mengharuskan anda untuk mengisi kebutuhan yang tercantum pada parameter diatas, dan hasilnya kemudian akan dikalkulasikan dalam bentuk persentase jumlah kadar lemak dalam tubuh kita.

### Goals
Untuk mengetahui kadar lemak dalam tubuh, sehingga bisa membantu memantau kondisi kesehatan.
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, dimana isi dari Body Fat Prediction Dataset ini yaitu hasil pengukuran lingkar tubuh 252 pria dan diteliti berdasarkan 14 atribut diatas.Dataset ini digunakan untuk menggambarkan teknik regresi berganda. Pengukuran lemak tubuh yang akurat tidaklah mudah dan diinginkan adanya metode yang mudah untuk memperkirakan lemak tubuh yang tidak merepotkan atau mahal.

[Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset/data). 

### Variabel-variabel pada Body Fat Prediction Dataset adalah sebagai berikut:
- Density :
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

#Visualisasi Data
visualisasi heatmap untuk melihat sebaran data pada dataset bodyfat.csv , masukan perintah :
```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
```bash
out : <img width="338" alt="Screenshot 2023-10-26 230013" src="https://github.com/khildafitria/machinelearning/assets/149028314/60ba8229-27ff-4895-a4f2-7f23b5730a3a">
```







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

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

![image](https://github.com/khildafitria/machinelearning/assets/149028314/ce9faa42-8b9e-4fda-aff6-97d2734cd341)

