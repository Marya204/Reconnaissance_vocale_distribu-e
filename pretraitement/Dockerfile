FROM python:3.11-slim

# Installer les dépendances système nécessaires à la compilation de PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'app
COPY . .

# Exposer le port (si Flask)
EXPOSE 5000

# Commande pour lancer ton app Flask (si tu l'utilises)
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

