import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# Mendapatkan path direktori tempat script Python berada
base_dir = os.path.dirname(__file__)

# Mendapatkan path ke file CSV menggunakan path relatif
file_name = 'Data_ISPU_DKI_JAKARTA_2018_2023.csv'
file_path = os.path.join(base_dir, file_name)

# Memuat data
df = pd.read_csv(file_path, sep=',', decimal=',')
print(df)

df.replace('-', np.nan, inplace=True)
df.replace('---', np.nan, inplace=True)

# Data Cleaning - Mengisi nilai NaN dengan rata-rata kolom
df.dropna(axis=0,inplace=True)

# Data Preprocessing
# Memilih fitur dan target
df_X = df.drop(['periode_data', 'tanggal', 'stasiun', 'max', 'parameter_pencemar_kritis', 'kategori'], axis=1)
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

# Membuat dan melatih model
model = RandomForestClassifier(
    n_estimators=100,           # Default is 100
    max_depth=None,             # Default is None (nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
    min_samples_split=3,        # Default is 2
    min_samples_leaf=2,         # Default is 1
    max_features='sqrt',        # Default is 'auto'
    bootstrap=True,             # Default is True
    random_state=42             # To ensure reproducibility
)
model.fit(X_train, y_train)

# Menyimpan model
joblib.dump(model, 'modelRN.pkl')

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