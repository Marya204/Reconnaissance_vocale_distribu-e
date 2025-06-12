from kfp import Client
from kfp import compiler
from pipeline import asr_pipeline  # Assure-toi que pipeline.py est dans le même répertoire

# Étape 1 : Compiler le pipeline en YAML
compiler.Compiler().compile(
    pipeline_func=asr_pipeline,
    package_path='asr_pipeline.yaml'
)

# Étape 2 : Créer un client pour envoyer le pipeline
client = Client()  # Tu peux ajouter host='http://<ton-kubeflow-ip>' si nécessaire

# Étape 3 : Charger et exécuter le pipeline à partir du fichier YAML
client.create_run_from_pipeline_package(
    pipeline_file='asr_pipeline.yaml',
    arguments={
        'audio_input': '/data/input.wav',   # Assure-toi que ces fichiers existent dans le container
        'model_path': '/data/model_dir'
    }
)
