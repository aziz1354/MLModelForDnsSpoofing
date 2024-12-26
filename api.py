from flask import Flask, request, jsonify
import subprocess
import sys
import dns.resolver
import socket
from dnsextraction import extract_dns_features

app = Flask(__name__)

def check_spamhaus(ip_address):
    try:
        # Reverse the IP address for Spamhaus query
        reversed_ip = '.'.join(reversed(ip_address.split('.')))
        query = f"{reversed_ip}.zen.spamhaus.org"
        
        # Perform DNS query
        dns.resolver.resolve(query, 'A')
        
        # If we get a response, the IP is in the Spamhaus list
        return 1
    except dns.resolver.NXDOMAIN:
        # If no response is received, the IP is not in the list
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1  # Return -1 to indicate an error

def is_valid_ip(ip):
    try:
        # Try to resolve the IP address
        socket.inet_pton(socket.AF_INET, ip)
        return True
    except socket.error:
        # If it raises an error, it's not a valid IP
        return False

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Check if IP address and domain are provided
    if 'ip' not in data or 'domain' not in data:
        return jsonify({'error': 'Both "ip" and "domain" parameters are required.'}), 400
    
    ip = data['ip']
    domain = data['domain']
    
   
    
    # Validate the IP address
    if not is_valid_ip(ip):
        return jsonify({'error': 'Invalid IP address format.'}), 400
    
    # Check Spamhaus list
    spamhaus_result = check_spamhaus(ip)
    
    # If IP is in Spamhaus malicious list
    if spamhaus_result == 1:
        return jsonify({'prediction': 'malicious'})
    
    # If Spamhaus check fails or returns 0, proceed with domain analysis
    try:
        # Extract DNS features
        features = extract_dns_features(domain)
        
        # Prepare ordered arguments for prediction script
        ordered_args = [
            '--FQDN_count', features['FQDN_count'],
            '--subdomain_length', features['subdomain_length'],
            '--upper', features['upper'],
            '--lower', features['lower'],
            '--numeric', features['numeric'],
            '--entropy', features['entropy'],
            '--special', features['special'],
            '--labels', features['labels'],
            '--labels_max', features['labels_max'],
            '--labels_average', features['labels_average'],
            '--len', features['len'],
            '--subdomain', features['subdomain'],
            '--longest_word', f"\"{features['longest_word']}\"",
            '--sld', f"\"{features['sld']}\""
        ]
        
        # Construct the command
        command = ['python', 'prediction.py'] + list(map(str, ordered_args))
        
        # Run prediction script
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': 'Error executing prediction.py', 'details': result.stderr}), 500
        
        
        output_lines = result.stdout.strip().split("\n")
        prediction = output_lines[-1].strip().replace("Le domaine est classifié comme: ", "")
        
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
