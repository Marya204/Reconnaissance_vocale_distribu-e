import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.asr_model import build_model

# --- Configuration ---
IMG_HEIGHT = 128
IMG_DIR = "data/spectrograms"
TRANSCRIPTION_PATH = "data/transcriptions.txt"
BATCH_SIZE = 2  
EPOCHS = 20
SAVED_MODEL_DIR = "models/saved_model"
alphabet = "abcdefghijklmnopqrstuvwxyz "
NUM_CLASSES = len(alphabet) + 1  # 27 + 1 pour le 'blank' de CTC

# --- Générateur efficace ---
def data_generator(image_paths, labels, input_lengths, label_lengths, batch_size, shuffle=True):
    indices = np.arange(len(image_paths))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_ids = indices[start:end]
            batch_images = []
            for path in image_paths[batch_ids]:
                img = Image.open(path).convert("L")  # grayscale
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)  # shape: (128, width, 1)
                batch_images.append(img)
            max_width = max(img.shape[1] for img in batch_images)
            padded_images = [np.pad(img, ((0, 0), (0, max_width - img.shape[1]), (0, 0))) for img in batch_images]
            batch_X = np.stack(padded_images)
            yield {
                'input': batch_X,
                'label': labels[batch_ids],
                'input_length': input_lengths[batch_ids].reshape(-1, 1),
                'label_length': label_lengths[batch_ids].reshape(-1, 1)
            }, {
                'ctc_loss': np.zeros((len(batch_ids), 1))
            }

# --- Entraînement ---
def train():
    # Lire les transcriptions
    image_paths = []
    labels = []
    with open(TRANSCRIPTION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            img_name, text = line.strip().split("|")
            image_paths.append(os.path.join(IMG_DIR, img_name))
            label = [alphabet.index(c) for c in text if c in alphabet]
            labels.append(label)

    label_seqs = pad_sequences(labels, padding='post')
    # Correction: use width (size[1]) for input length, not height (size[0])
    input_lengths = np.array([Image.open(p).size[1] // 4 for p in image_paths])  # ajuste si ton réseau a plus de pooling
    label_lengths = np.array([len(l) for l in labels])

    image_paths = np.array(image_paths)
    label_seqs = np.array(label_seqs)
    input_lengths = np.array(input_lengths)
    label_lengths = np.array(label_lengths)

    # Split
    X_train, X_val, y_train, y_val, len_train, len_val, label_len_train, label_len_val = train_test_split(
        image_paths, label_seqs, input_lengths, label_lengths, test_size=0.2, random_state=42
    )

    # Filter out examples where label_length > input_length - 10
    def filter_valid(X, y, in_len, lab_len):
        valid = [i for i in range(len(X)) if lab_len[i] <= in_len[i] - 10]
        return X[valid], y[valid], in_len[valid], lab_len[valid]

    X_train, y_train, len_train, label_len_train = filter_valid(X_train, y_train, len_train, label_len_train)
    X_val, y_val, len_val, label_len_val = filter_valid(X_val, y_val, len_val, label_len_val)

    print("Données :")
    print(f"  - {len(X_train)} exemples d'entraînement")
    print(f"  - {len(X_val)} exemples de validation")
    print("Exemples (input_length, label_length):")
    for i in range(min(10, len(X_train))):
        print(f"  {len_train[i]} (input) / {label_len_train[i]} (label)")

    # Construction du modèle
    base_model = build_model(input_shape=(IMG_HEIGHT, None, 1), num_classes=NUM_CLASSES)
    print("Sortie du modèle :", base_model.output_shape)

    # CTC loss inputs
    labels_input = tf.keras.Input(name='label', shape=(None,), dtype='int32')
    input_length_input = tf.keras.Input(name='input_length', shape=(1,), dtype='int32')
    label_length_input = tf.keras.Input(name='label_length', shape=(1,), dtype='int32')

    # Perte CTC
    loss_out = tf.keras.layers.Lambda(
        lambda args: K.ctc_batch_cost(*args), name='ctc_loss'
    )([labels_input, base_model.output, input_length_input, label_length_input])

    model = tf.keras.Model(
        inputs=[base_model.input, labels_input, input_length_input, label_length_input],
        outputs=loss_out
    )

    model.compile(optimizer='adam', loss={'ctc_loss': lambda y_true, y_pred: y_pred})

    # Callbacks
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
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

    # Entraînement
    model.fit(
        data_generator(X_train, y_train, len_train, label_len_train, BATCH_SIZE),
        validation_data=data_generator(X_val, y_val, len_val, label_len_val, BATCH_SIZE, shuffle=False),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        validation_steps=max(1, len(X_val) // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Sauvegarde du modèle de base
    base_model.save(os.path.join(SAVED_MODEL_DIR, "final_asr_model.keras"))
    print(f"Modèle sauvegardé dans : {SAVED_MODEL_DIR}/final_asr_model.keras")


if __name__ == "__main__":
    train()
