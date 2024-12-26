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

# Identifier les colonnes numériques et catégorielles
numerical_features = [
    'FQDN_count', 'subdomain_length', 'upper', 'lower', 
    'numeric', 'entropy', 'special', 'labels', 
    'labels_max', 'labels_average', 'len', 'subdomain'
]

categorical_features = ['longest_word', 'sld']

# Combiner et nettoyer les données
combined_data = pd.concat([benign_data, malicious_data], ignore_index=True)

# Gérer les valeurs manquantes
combined_data['longest_word'] = combined_data['longest_word'].fillna('unknown')

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

# Sauvegarder les données de test dans un fichier CSV
test_data = X_test.copy()
test_data['class'] = y_test  # Ajouter la colonne 'class' aux données de test

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

# Ajouter les résultats des prédictions aux données de test
test_data['predicted_class'] = y_pred  # Ajouter la colonne 'predicted_class'

# Sauvegarder les données de test avec les prédictions dans un fichier CSV
test_data.to_csv('test_data_with_predictions.csv', index=False)  # Sauvegarder dans un fichier CSV

# Afficher un message de confirmation
print("Les données de test avec les prédictions ont été sauvegardées dans 'test_data_with_predictions.csv'.")

# Évaluer le modèle
print("\nRapport de Classification:")
print(classification_report(y_test, y_pred))

print("\nMatrice de Confusion:")
print(confusion_matrix(y_test, y_pred))

# Fonction de prédiction
def predict_dns_domain(domain_features):
    """
    Prédit si un domaine est bénin ou malveillant
    """
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame([domain_features])
    
    # Encoder les features catégorielles
    for cat_feature, le in label_encoders.items():
        # Remplacer les valeurs inconnues par 'unknown'
        df[cat_feature] = df[cat_feature].apply(lambda x: x if x in le.classes_ else 'unknown')
        df[cat_feature] = le.transform(df[cat_feature].astype(str))
    
    # Standardiser les features
    df_scaled = df.copy()
    df_scaled[numerical_features] = scaler.transform(df[numerical_features])
    
    # Sélectionner les features
    df_selected = selector.transform(df_scaled)
    
    return classifier.predict(df_selected)[0]

# Exemple d'utilisation
example_domain = {
    'FQDN_count': 1,
    'subdomain_length': 10,
    'upper': 2,
    'lower': 8,
    'numeric': 0,
    'entropy': 3.5,
    'special': 1,
    'labels': 2,
    'labels_max': 15,
    'labels_average': 7.5,
    'len': 20,
    'subdomain': 1,
    'longest_word': 'example',  # Vous pouvez remplacer cette valeur par une nouvelle valeur inconnue
    'sld': 'test'
}

print("\nPrédiction pour un domaine exemple:")
print("Classe prédite:", predict_dns_domain(example_domain))

# Sauvegarder le modèle
joblib.dump({
    'classifier': classifier,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'selector': selector,
}, 'dns_classification_model.pkl')
print("\nModèle sauvegardé avec succès.")
