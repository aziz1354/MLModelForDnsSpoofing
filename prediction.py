import pandas as pd
import joblib
import argparse

# Charger le modèle et les objets sauvegardés
model_data = joblib.load('dns_classification_model.pkl')
classifier = model_data['classifier']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
selector = model_data['selector']

# Liste des features numériques et catégorielles
numerical_features = [
    'FQDN_count', 'subdomain_length', 'upper', 'lower', 
    'numeric', 'entropy', 'special', 'labels', 
    'labels_max', 'labels_average', 'len', 'subdomain'
]

categorical_features = ['longest_word', 'sld']

# Fonction de prédiction
def predict_dns_domain(domain_features):
    """
    Prédit si un domaine est bénin ('not malicious') ou malveillant ('malicious')
    """
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame([domain_features])
    print("Données brutes (avant encodage) :")
    print(df)
    
    # Encoder les features catégorielles et gérer les valeurs numériques
    for cat_feature, le in label_encoders.items():
        if cat_feature in df.columns:
            # Vérifier si la feature est numérique ou catégorielle
            if df[cat_feature].dtype == 'object' or isinstance(df[cat_feature].iloc[0], str):
                # Gérer les valeurs inconnues sans les transformer en 'unknown'
                df[cat_feature] = df[cat_feature].apply(lambda x: x if x in le.classes_ else None)
                # Transformer les valeurs avec le LabelEncoder, en gérant les valeurs manquantes
                df[cat_feature] = df[cat_feature].apply(lambda x: le.transform([x])[0] if x is not None else -1)  # Utiliser -1 pour les valeurs inconnues
            else:
                # Si la feature est numérique, la laisser telle quelle
                df[cat_feature] = df[cat_feature].astype(float)  # Assurez-vous que la feature est bien au format numérique
    print("Données après encodage :")
    print(df)
    
    # Standardiser les features
    df_scaled = df.copy()
    df_scaled[numerical_features] = scaler.transform(df[numerical_features])
    
    print("Données après standardisation :")
    print(df_scaled)
    
    # Sélectionner les features
    df_selected = selector.transform(df_scaled)
    
    print("Données après sélection des features :")
    print(df_selected)
    
    # Faire la prédiction
    prediction = classifier.predict(df_selected)[0]
    
    # Retourner le résultat sous forme lisible
    return 'malicious' if prediction == 1 else 'not malicious'

# Fonction pour récupérer les paramètres à partir du terminal
def parse_arguments():
    parser = argparse.ArgumentParser(description='Prédiction de domaine DNS')
    for feature in numerical_features + categorical_features:
        parser.add_argument(f'--{feature}', type=str, required=True, help=f'Valeur de la feature {feature}')
    return parser.parse_args()

# Fonction principale
def main():
    args = parse_arguments()
    
    # Convertir les arguments en dictionnaire
    domain_features = {key: args.__dict__[key] for key in args.__dict__}
    
    # Prédiction pour les données fournies
    prediction = predict_dns_domain(domain_features)
    
    if prediction:
        print(f"\nLe domaine est classifié comme: {prediction}")
    else:
        print("Erreur lors de la prédiction.")

if __name__ == '__main__':
    main()
