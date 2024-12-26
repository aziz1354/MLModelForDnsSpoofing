import subprocess
import tldextract
import re
import math
from collections import Counter
import argparse

# Fonction pour calculer l'entropie d'une chaîne de caractères
def calculate_entropy(s):
    if len(s) == 0:
        return 0
    freq = Counter(s)
    probs = [freq[char] / len(s) for char in freq]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy


# Fonction pour encoder une valeur catégorique (sld ou longest_word)
def encode_value(encoder_script, model_path, field, value):
    try:
        result = subprocess.run(
            ["python", encoder_script, model_path, field, value],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())  # On suppose que le script retourne l'encodage en entier
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {encoder_script} : {e}")
        return None


# Fonction principale pour extraire les caractéristiques
def extract_dns_features(domain, encoder_script="encoder.py", model_path="dns_classification_model.pkl"):
    # Extraire les parties du domaine avec tldextract
    extracted = tldextract.extract(domain)
    
    # Extraire les sous-domaines, le domaine de second niveau (SLD), et le suffixe
    subdomains = extracted.subdomain.split('.') if extracted.subdomain else []
    domain_name = extracted.domain
    suffix = extracted.suffix
    
    # Caractéristiques numériques
    FQDN_count = len(subdomains) + 1  # Nombre total de labels (y compris le domaine et suffixe)
    subdomain_count = len(subdomains)  # Nombre de sous-domaines
    subdomain_length = sum(len(sub) for sub in subdomains)  # Longueur totale des sous-domaines
    upper = sum(1 for c in domain if c.isupper())  # Nombre de lettres majuscules
    lower = sum(1 for c in domain if c.islower())  # Nombre de lettres minuscules
    numeric = sum(1 for c in domain if c.isdigit())  # Nombre de chiffres
    special = sum(1 for c in domain if not c.isalnum() and c != '.')  # Nombre de caractères spéciaux
    entropy = calculate_entropy(domain)  # Entropie du domaine
    
    # Nombre total de labels
    labels = len(subdomains) + 1  # Nombre total de labels (y compris domaine et suffixe)
    
    # Longueur des labels
    labels_max = max(len(label) for label in subdomains + [domain_name, suffix]) if subdomains else 0
    labels_average = sum(len(label) for label in subdomains + [domain_name, suffix]) / len(subdomains + [domain_name, suffix]) if subdomains else 0
    
    # Le mot le plus long dans le domaine
    longest_word = max(re.findall(r'\w+', domain), key=len) if domain else ""
    
    # Domaine de second niveau (SLD)
    sld = domain_name  # Le domaine de second niveau est simplement la partie avant le TLD
    
    # Longueur totale du domaine
    domain_length = len(domain)
    
    # Encodage des champs catégoriques
   
    
    
    
    # Résultat sous forme de dictionnaire
    features = {
        'FQDN_count': FQDN_count,
        'subdomain_length': subdomain_length,
        'upper': upper,
        'lower': lower,
        'numeric': numeric,
        'entropy': entropy,
        'special': special,
        'labels': labels,
        'labels_max': labels_max,
        'labels_average': labels_average,
        'longest_word': longest_word,
        'sld': sld,
        'len': domain_length,
        'subdomain': subdomain_count
    }
    
    return features


# Fonction pour analyser les arguments en ligne de commande
def main():
    # Initialiser l'argument parser
    parser = argparse.ArgumentParser(description="Extraire les caractéristiques d'un domaine DNS")
    parser.add_argument("domain", help="Nom du domaine DNS à analyser")
    parser.add_argument("--encoder", default="encoder.py", help="Script d'encodage des catégories")
    parser.add_argument("--model", default="dns_classification_model.pkl", help="Chemin vers le modèle d'encodage")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Extraire les caractéristiques du domaine
    domain = args.domain
    features = extract_dns_features(domain, args.encoder, args.model)
    
    # Afficher les caractéristiques
    print(f"Caractéristiques du domaine '{domain}':")
    for key, value in features.items():
        print(f"{key}: {value}")


# Exécution du script
if __name__ == "__main__":
    main()
