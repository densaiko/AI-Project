**Overview Materi Training Astra Honda Motor: Data Science & AI (5 Hari)**

- Classification ML Deployment [![Classification ML Deployment](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/spaces/densaiko/EmployeePromotionPrediction)
- Regression ML Deployment [![Regression ML Deployment](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/spaces/densaiko/AHMPredictCar)
- RAG AI Deployment [![RAG AI Deployment](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/spaces/densaiko/AstraHondaMotorV2)

---

### **Day 1: Python for AI & Data Preprocessing**

**Topik Inti:**

- Python dasar untuk data science
- Data cleaning (missing values, outlier)
- Encoding data kategorikal
- Scaling & normalization

**Deskripsi:** Hari pertama bertujuan membekali peserta dengan keterampilan dasar pemrograman Python untuk keperluan data science, sekaligus memahami bagaimana data mentah perlu diproses terlebih dahulu sebelum dimasukkan ke dalam model machine learning. Proses pembersihan dan transformasi data sangat krusial karena kualitas data sangat memengaruhi akurasi model.

**1. Python Dasar** Peserta belajar struktur dasar seperti fungsi, kondisi, perulangan, serta menerapkan fungsi `apply` dan `try-except` dalam konteks data tabular. Hal ini penting untuk melakukan transformasi kolom dan menangani anomali.

```python
def cek_status(nilai):
    if nilai <= 2010:
        return 'Brand Lama'
    return 'Brand Baru'

df['status'] = df['year'].apply(cek_status)
```

**2. Data Cleaning – Missing Values** Missing value (nilai kosong/NaN) sering terjadi akibat kesalahan input atau data tidak lengkap. Nilai ini perlu ditangani karena sebagian besar algoritma tidak mendukung data kosong. Pengisian dapat dilakukan dengan rata-rata, median, modus, atau nilai konstan.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['kolom']] = imputer.fit_transform(df[['kolom']])
```

**3. Outlier Detection** Outlier adalah nilai yang jauh dari pola umum dan dapat memengaruhi distribusi data serta performa model. Visualisasi seperti boxplot membantu mendeteksi outlier.

```python
import seaborn as sns
sns.boxplot(data=df, x='price')
```

**4. Encoding Data Kategorikal** Model ML hanya memahami angka, sehingga kolom dengan tipe data string perlu dikonversi ke format numerik. Dua metode umum adalah Label Encoding dan One-Hot Encoding. Pemilihan metode bergantung pada jenis model yang digunakan dan jumlah kategori.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['fuel_encoded'] = le.fit_transform(df['fuel'])
```

**5. Scaling & Normalization** Skala fitur dapat sangat berbeda, misalnya pendapatan vs umur. Scaling menstandarkan data agar memiliki mean 0 dan standar deviasi 1. Normalisasi menyetarakan data dalam rentang 0–1. Ini penting terutama untuk model berbasis jarak seperti SVM dan KNN.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['engine_size']] = scaler.fit_transform(df[['engine_size']])
```

---

### **Day 2: Exploratory Data Analysis & Classification**

**Topik Inti:**

- Visualisasi data (bar chart, boxplot, heatmap, pie chart)
- Univariate dan bivariate analysis
- Konsep klasifikasi dan evaluasi model

**Deskripsi:** EDA adalah proses eksplorasi awal terhadap data dengan tujuan mengenal distribusi nilai, mendeteksi outlier, dan memahami hubungan antar fitur. Hal ini membantu dalam pemilihan fitur, transformasi data, dan deteksi masalah seperti multikolinearitas.

**1. Univariate & Bivariate Analysis** Univariate fokus pada satu fitur seperti distribusi umur, sedangkan bivariate melihat hubungan dua fitur seperti korelasi antara umur dan pendapatan. Visualisasi digunakan untuk mempermudah interpretasi.

```python
sns.histplot(df['age'])  # univariate
sns.scatterplot(data=df, x='age', y='income')  # bivariate
```

**2. Visualisasi Data** Visualisasi mempermudah pemahaman terhadap data dan tren. Bar chart cocok untuk data kategorikal, boxplot untuk distribusi numerik, heatmap untuk korelasi antar fitur, dan pie chart untuk proporsi kelompok.

```python
sns.boxplot(data=df, x='gender', y='income')
sns.heatmap(df.corr(), annot=True)
```

**3. Model Klasifikasi** Model klasifikasi digunakan saat target berupa kategori (misalnya: spam/tidak spam). Peserta dikenalkan dengan Logistic Regression (linear), Decision Tree (berbasis aturan), dan Random Forest (ansambel).

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

**4. Evaluasi Model Klasifikasi** Penggunaan Confusion Matrix dan metrik seperti Accuracy, Precision, Recall, dan F1 Score penting untuk mengetahui kualitas model, khususnya saat data imbalance.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

### **Day 3: Regression & Hyperparameter Tuning**

**Topik Inti:**

- Simple & Multiple Linear Regression
- Evaluasi model regresi (MAE, MSE, R2)
- K-Fold Cross Validation & GridSearchCV

**Deskripsi:** Regresi digunakan ketika target berupa nilai kontinu (numerik), seperti harga, suhu, atau pendapatan. Peserta belajar membedakan antara simple linear regression (satu input) dan multiple linear regression (multi input). Proses ini memperkenalkan konsep prediksi berbasis relasi linier antar variabel.

**1. Simple & Multiple Regression**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

**2. Evaluasi Model Regresi**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**3. Hyperparameter Tuning (GridSearchCV)**

```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
gs = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5)
gs.fit(X_train, y_train)
```

---

### **Day 4: NLP Text Classification & LLM-RAG Introduction**

**Topik Inti:**

- Tokenization, stopwords, stemming, TF-IDF
- Text classification dengan ML
- Pengenalan LLM (Large Language Models) & RAG (Retrieval-Augmented Generation)

**Deskripsi:** Hari keempat berfokus pada data teks. Peserta belajar mengubah kalimat menjadi representasi numerik yang bisa dipahami oleh algoritma machine learning. Kemudian diperkenalkan konsep dasar LLM dan bagaimana mereka digunakan dalam Retrieval-Augmented Generation (RAG).

**1. Tokenization & Stopwords**

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "Mobil listrik ramah lingkungan"
tokens = [w for w in word_tokenize(text) if w not in stopwords.words('indonesian')]
```

**2. Stemming & Lemmatization**

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()
print(stemmer.stem("mengantarkan"))  # hasil: antar
```

**3. TF-IDF Vectorization**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(corpus)
```

**4. Pengenalan LLM & RAG** Large Language Models (seperti GPT-4) dilatih dengan dataset besar dan mampu memahami konteks panjang. RAG memperkuat kemampuan LLM dengan menggabungkan dokumen eksternal menggunakan vector search (Chroma, FAISS), lalu hasilnya digunakan sebagai konteks oleh LLM.

---

### **Day 5: Model Deployment**

**Topik Inti:**

- Konsep API dan HTTP response
- Membuat REST API dengan Flask
- Postman & Testing API
- Model serialization (pickle)

**Deskripsi:** Pada hari terakhir, peserta diajak memahami bagaimana model machine learning bisa digunakan dalam aplikasi nyata melalui API. Ini mencakup proses menyimpan model, membuat server API, dan menguji endpoint menggunakan Postman.

**1. Membuat API Sederhana dengan Flask**

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify(prediction.tolist())
```

**2. Testing API via Postman** Postman digunakan untuk menguji endpoint API secara manual. Peserta dapat mengirim data JSON dan melihat apakah API merespons dengan benar.

**3. Model Serialization (Pickle)**

```python
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**4. Struktur umum proyek deployment ML:**

- `app.py`: tempat deklarasi endpoint dan logic API
- `model.py`: memuat dan menjalankan model
- `eda.py`: script tambahan untuk eksplorasi/analisis data (opsional)

---

Dokumen ini dirancang sebagai ringkasan esensial untuk membantu peserta memahami konteks dari setiap hari pelatihan secara terstruktur, lengkap, dan praktis.

