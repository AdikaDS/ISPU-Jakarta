import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Mendapatkan path direktori tempat script Python berada
base_dir = os.path.dirname(__file__)

# Mendapatkan path ke file CSV menggunakan path relatif
file_name = 'Data_ISPU_DKI_JAKARTA.csv'
file_path = os.path.join(base_dir, file_name)

# Memuat data
df = pd.read_csv(file_path, sep=',', decimal=',')

# Menampilkan nama kolom
# print(df)

# Data Cleaning - Mengisi nilai NaN dengan rata-rata kolom
df = df.dropna()

# columns_with_nan = ["pm_duakomalima", "pm_sepuluh", "sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"]
# for column in columns_with_nan:
#     if column in df.columns:
#         mean_value = df[column].mean()
#         df[column].fillna(mean_value, inplace=True)
#     else:
#         print(f"Kolom {column} tidak ditemukan dalam dataset.")

# Data Preprocessing
# Memilih fitur dan target
df_X = df.drop(['periode_data', 'bulan', 'tanggal', 'stasiun', 'max', 'parameter_pencemar_kritis', 'kategori'], axis=1)
df_y = df[['kategori']]

# Label Encoding untuk target
le = LabelEncoder()
df_y = le.fit_transform(df_y['kategori'])
# Mengubah X dan Y menjadi array NumPy
X = df_X.astype(float).values
y = df_y.astype(float)

# Membagi data menjadi 80% Training dan 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Menyimpan Scaler
# joblib.dump(scaler, 'scaler.pkl')

# Membuat dan melatih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Menyimpan model
joblib.dump(model, 'model.pkl')

# Menyimpan Label Encoder
# joblib.dump(le, 'label_encoder.pkl')

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')