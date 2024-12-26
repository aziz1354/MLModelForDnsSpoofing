import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Charger les datasets
benign_data = pd.read_csv('stateless_features-benign_1.pcap.csv', sep=',')
malicious_data = pd.read_csv('stateless_features-heavy_compressed.pcap.csv', sep=',')

# Ajouter une colonne de classe à chaque dataset
benign_data['class'] = 0
malicious_data['class'] = 1

# Identifier la taille du dataset malicious
malicious_size = malicious_data.shape[0]

# Sélectionner une sous-partie du dataset benign de taille équivalente
benign_data_balanced = benign_data.sample(n=malicious_size, random_state=42)

# Combiner les deux datasets pour un dataset équilibré
combined_data = pd.concat([benign_data_balanced, malicious_data], ignore_index=True)

# Mélanger les données pour éviter tout biais
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Identifier les features numériques et catégorielles
numerical_features = [
    'FQDN_count', 'subdomain_length', 'upper', 'lower', 
    'numeric', 'entropy', 'special', 'labels', 
    'labels_max', 'labels_average', 'len', 'subdomain'
]

categorical_features = ['longest_word', 'sld']

# Sélectionner et préparer les features
X = combined_data[numerical_features + categorical_features]
y = combined_data['class']

# Encoder les features catégorielles
label_encoders = {}
for cat_feature in categorical_features:
    le = LabelEncoder()
    X[cat_feature] = le.fit_transform(X[cat_feature].astype(str))
    label_encoders[cat_feature] = le

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les features numériques
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# Sélection de features
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Entraîner le classificateur naïf bayésien
classifier = GaussianNB()
classifier.fit(X_train_selected, y_train)

# Prédictions pour les données de test
y_pred = classifier.predict(X_test_selected)

# Afficher le rapport de classification et la matrice de confusion
print("\nRapport de Classification:")
print(classification_report(y_test, y_pred))

print("\nMatrice de Confusion:")
print(confusion_matrix(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump({
    'classifier': classifier,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'selector': selector,
}, 'dns_classification_model_balanced.pkl')
print("\nModèle sauvegardé avec succès.")
