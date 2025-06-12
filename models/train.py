import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.asr_model import build_model
from controllers.prepare_data import load_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
LABELS_PATH = os.path.join(BASE_DIR, "data", "transcriptions.txt")

# --- Configurations ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
LABELS_PATH = os.path.join(BASE_DIR, "data", "transcriptions.txt")  # Chemin corrigé
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_model")
# Ajoutez ce debug utile avant l'entraînement :
print(f"Vérification des chemins:")
print(f"- Spectrogrammes: {DATA_DIR} (existe: {os.path.exists(DATA_DIR)})")
print(f"- Transcripts: {LABELS_PATH} (existe: {os.path.exists(LABELS_PATH)})")
print(f"- Contenu data/: {os.listdir(os.path.join(BASE_DIR, 'data')) if os.path.exists(os.path.join(BASE_DIR, 'data')) else 'Dossier data/ introuvable'}")
# Assurez-vous que le dossier existe
import os
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
BATCH_SIZE = 16
EPOCHS = 10
IMG_HEIGHT, IMG_WIDTH = 128, 128
char_to_num = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}
NUM_CLASSES =  len(char_to_num) + 1  # +1 pour le blank

# --- Data generator ---
def data_generator(images, labels, input_lengths, label_lengths, batch_size, shuffle=True):
    indices = np.arange(len(images))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(images), batch_size):
            batch_ids = indices[i:i+batch_size]
            batch_imgs = images[batch_ids]
            batch_labels = labels[batch_ids]
            batch_input_lengths = input_lengths[batch_ids].reshape(-1, 1)
            batch_label_lengths = label_lengths[batch_ids].reshape(-1, 1)

            inputs = {
                'input': batch_imgs,
                'label': batch_labels,
                'input_length': batch_input_lengths,
                'label_length': batch_label_lengths
            }
            outputs = np.zeros((len(batch_imgs), 1))
            yield inputs, outputs


# --- Entraînement principal ---
def train():
    # Modèle de base
    base_model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)

    # Calcul du nombre de time steps en sortie du modèle avant la couche CTC
    # Par exemple, si la sortie est (batch_size, time_steps, features), on récupère time_steps
    # Pour ça, on peut créer un modèle partiel de l'entrée à la couche avant la sortie finale
    # Ou vérifier directement la sortie du modèle (sans la couche CTC)

    # Supposons que la sortie est (None, time_steps, num_classes), on récupère time_steps
    time_steps = base_model.output_shape[1]

    # Chargement des données avec time_steps en input_length
    spectrograms, labels, input_lengths, label_lengths = load_data(DATA_DIR, LABELS_PATH, img_height=IMG_HEIGHT, img_width=IMG_WIDTH
    )


    # Split
    X_train, X_val, y_train, y_val, len_train, len_val, label_len_train, label_len_val = train_test_split(
        spectrograms, labels, input_lengths, label_lengths, test_size=0.2, random_state=42
    )

    print("Temps d'entrée attendu (time steps) :", base_model.output_shape[1])

    # Inputs supplémentaires pour le CTC
    labels_input = tf.keras.Input(name='label', shape=(None,), dtype='int32')
    input_length_input = tf.keras.Input(name='input_length', shape=(1,), dtype='int32')
    label_length_input = tf.keras.Input(name='label_length', shape=(1,), dtype='int32')

    # CTC Loss Lambda
    loss_out = tf.keras.layers.Lambda(
        lambda args: K.ctc_batch_cost(*args), name='ctc_loss')(
        [labels_input, base_model.output, input_length_input, label_length_input]
    )

    # Modèle final
    model = tf.keras.Model(
        inputs=[base_model.input, labels_input, input_length_input, label_length_input],
        outputs=loss_out
    )

    model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

    # Générateurs
    train_gen = data_generator(X_train, y_train, len_train, label_len_train, BATCH_SIZE)
    val_gen = data_generator(X_val, y_val, len_val, label_len_val, BATCH_SIZE, shuffle=False)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVED_MODEL_DIR, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_steps=len(X_val) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nPréparation du modèle pour la production...")
    inference_model = tf.keras.Model(
        inputs=base_model.input, 
        outputs=base_model.output
    )
    
    print(f"\nSauvegarde dans {SAVED_MODEL_DIR}")
    inference_model.save(os.path.join(SAVED_MODEL_DIR, "saved_model.keras"))

    
    # Vérification (ajouter 5-10 lignes après)
    saved_files = os.listdir(SAVED_MODEL_DIR)
    assert 'saved_model.pb' in saved_files, "ERREUR: saved_model.pb manquant!"
    assert 'variables' in saved_files, "ERREUR: dossier variables manquant!"
    print("✅ Modèle correctement sauvegardé")

if __name__ == "__main__":
    train()
