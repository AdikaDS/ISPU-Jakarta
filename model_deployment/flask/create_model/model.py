import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

# Mendapatkan path direktori tempat script Python berada
base_dir = os.path.dirname(__file__)

# Mendapatkan path ke file CSV menggunakan path relatif
file_name = 'Data_ISPU_DKI_JAKARTA.csv'
file_path = os.path.join(base_dir, file_name)

# Memuat data
df = pd.read_csv(file_path, sep=',', decimal=',')

# df_cleaned = df[(df['kategori'] != 'SANGAT TIDAK SEHAT') & (df['kategori']!= 'TIDAK ADA DATA')]
df_cleaned = df[(df['kategori']!= 'TIDAK ADA DATA')]
df = df_cleaned

# Data Cleaning - Mengisi nilai NaN dengan rata-rata kolom
df['pm_sepuluh'].fillna(df['pm_sepuluh'].median(), inplace=True)
df['pm_duakomalima'].fillna(df['pm_duakomalima'].mean(), inplace=True)
df['sulfur_dioksida'].fillna(df['sulfur_dioksida'].median(), inplace=True)
df['karbon_monoksida'].fillna(df['karbon_monoksida'].median(), inplace=True)
df['nitrogen_dioksida'].fillna(df['nitrogen_dioksida'].median(), inplace=True)

df['parameter_pencemar_kritis'].fillna(df['parameter_pencemar_kritis'].value_counts().index[0], inplace=True)

# Data Preprocessing
# Memilih fitur dan target
df_X = df.drop(['periode_data', 'bulan', 'tanggal', 'stasiun', 'max', 'parameter_pencemar_kritis', 'kategori'], axis=1)
df_y = df[['kategori']]

# Label Encoding untuk target
le = LabelEncoder()
df_y = le.fit_transform(df_y['kategori'])

#categorical encoding
#merubah categorical value menjadi numerical value
#bisa pakai label encoding, ordinal atau one hot encoding
cats = df_X.select_dtypes(include=['object', 'bool']).columns
cat_features = list(cats.values)
for i in cat_features:
  le.fit(df_X[i])
  df_X[i] = le.transform(df_X[i])

# Mengubah X dan Y menjadi array NumPy
X = df_X.astype(float).values
y = df_y.astype(int)

# Membagi data menjadi 70% Training dan 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Periksa distribusi kelas sebelum resampling
print("Distribusi kelas sebelum resampling:", np.bincount(y_train))

# Mengatasi Imbalanced Classes menggunakan RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)


# Periksa distribusi kelas setelah resampling
print("Distribusi kelas setelah resampling:", np.bincount(y_train))

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

# Menyimpan Scaler
joblib.dump(scaler, 'scaler.pkl')

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