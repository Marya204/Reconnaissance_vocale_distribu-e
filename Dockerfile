# Étape 1 : image de base Python
FROM python:3.11-slim

# Étape 2 : Installer les dépendances système (notamment pour PyAudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Étape 3 : Définir le répertoire de travail
WORKDIR /app

# Étape 4 : Copier les dépendances
COPY requirements.txt .

# Étape 5 : Mettre à jour pip
RUN python -m pip install --upgrade pip

# Étape 6 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 7 : Copier l'ensemble du code (prétraitement + modèle)
COPY . .

# Étape 8 : Exposer le port FastAPI (Uvicorn)
EXPOSE 5000

# Étape 9 : Définir les variables d'environnement (optionnel avec FastAPI)
ENV PYTHONUNBUFFERED=1

# Étape 10 : Commande de démarrage de l'application FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=5000"]
