# Utiliser une image Python officielle
FROM python:3.11-slim

# Créer un répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY pipeline_app/ /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Spécifier la commande pour démarrer Flask
CMD ["python", "main.py"]
