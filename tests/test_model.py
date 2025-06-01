import os
from controllers.asr_controller import ASRController

def test_predict_output():
    # Utiliser un spectrogramme de test (fichier PNG généré à l'avance)
    test_image_path = 'static/spectrogramme.png'  # Assure-toi que ce fichier existe

    # Vérifier que le fichier existe
    assert os.path.exists(test_image_path), "Le fichier de test n'existe pas"

    # Appel du modèle
    model = ASRController()
    text, confidence = model.predict(test_image_path)

    # Vérifier que la sortie n'est pas vide
    assert isinstance(text, str), "La sortie texte doit être une chaîne de caractères"
    assert len(text.strip()) > 0, "Le texte transcrit est vide"
    assert confidence is None or (0 <= confidence <= 1), "La confiance doit être entre 0 et 1 ou None"
