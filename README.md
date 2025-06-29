**Overview Materi Training Astra Honda Motor: Data Science & AI (5 Hari)**

---

### **Day 1: Python for AI & Data Preprocessing**

**Topik Inti:**
- Python dasar untuk data science
- Data cleaning (missing values, outlier)
- Encoding data kategorikal
- Scaling & normalization

**1. Python Dasar**

```python
def cek_status(nilai):
    if nilai <= 2010:
        return 'Brand Lama'
    return 'Brand Baru'

df['status'] = df['year'].apply(cek_status)
```

**2. Data Cleaning – Missing Values**

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['kolom']] = imputer.fit_transform(df[['kolom']])
```

**3. Outlier Detection**

```python
import seaborn as sns
sns.boxplot(data=df, x='price')
```

**4. Encoding Data Kategorikal**

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['fuel_encoded'] = le.fit_transform(df['fuel'])
```

**5. Scaling & Normalization**

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

```python
sns.histplot(df['age'])  # univariate
sns.scatterplot(data=df, x='age', y='income')  # bivariate

sns.boxplot(data=df, x='gender', y='income')
sns.heatmap(df.corr(), annot=True)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

### **Day 3: Regression & Hyperparameter Tuning**

**Topik Inti:**
- Simple & Multiple Linear Regression
- Evaluasi model regresi (MAE, MSE, R2)
- K-Fold Cross Validation & GridSearchCV

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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
- Pengenalan LLM & RAG

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "Mobil listrik ramah lingkungan"
tokens = [w for w in word_tokenize(text) if w not in stopwords.words('indonesian')]

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()
print(stemmer.stem("mengantarkan"))

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(corpus)
```

---

### **Day 5: Model Deployment**

**Topik Inti:**
- Konsep API dan HTTP response
- REST API dengan Flask
- Postman & Testing API
- Model serialization (pickle)

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify(prediction.tolist())

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

Dokumen ini dirancang sebagai README pendamping training AI dan ML di Astra Honda Motor.
