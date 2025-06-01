import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image

# --- Configuration ---
DATASETS = {
    "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
}

BASE_DIR = "datasets"
INPUT_DIR = os.path.join(BASE_DIR, "LibriSpeech/dev-clean")
OUTPUT_DIR = "data/spectrograms"
TRANSCRIPT_FILE = "data/transcriptions.txt"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Téléchargement avec barre de progression ---
def download_url(url, output_path):
    with urllib.request.urlopen(url) as response:
        total = int(response.info().get("Content-Length").strip())
        with tqdm(total=total, unit="B", unit_scale=True, desc=output_path) as pbar:
            with open(output_path, 'wb') as f:
                while True:
                    buffer = response.read(1024 * 8)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

# --- Extraction .tar.gz ---
def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)

# --- Sauvegarde de spectrogrammes ---
def save_mel_spectrogram(wav_path, output_path):
    y, sr = sf.read(wav_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    plt.imshow(mel_db, cmap='gray')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- Nettoyage des transcriptions (enlève apostrophes) ---
def clean_transcriptions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            img, text = line.strip().split("|")
            cleaned = text.replace("'", "")  # enlève apostrophes
            f.write(f"{img}|{cleaned}\n")

# --- Préparation du dataset complet ---
def prepare_dataset():
    for name, url in DATASETS.items():
        archive_path = os.path.join(BASE_DIR, f"{name}.tar.gz")
        if not os.path.exists(archive_path):
            print(f"Téléchargement de {name}...")
            download_url(url, archive_path)
        else:
            print(f"{name}.tar.gz déjà présent.")

        if not os.path.exists(INPUT_DIR):
            print("Extraction...")
            extract_tar(archive_path, BASE_DIR)

    # Génération des spectrogrammes + transcriptions
    transcripts = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".flac"):
                full_path = os.path.join(root, file)
                base_name = Path(file).stem
                out_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
                save_mel_spectrogram(full_path, out_path)

            elif file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, text = parts
                            transcripts.append(f"{audio_id}.png|{text.lower()}")

    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        for line in transcripts:
            f.write(line + "\n")

    clean_transcriptions(TRANSCRIPT_FILE)
    print("✅ Préparation terminée. Les spectrogrammes sont dans `data/spectrograms`.")


def load_data(spectrogram_dir=OUTPUT_DIR, transcript_file=TRANSCRIPT_FILE,
              img_height=128, img_width=128):

    reduction_factor = 4  # Remplace 2 par 4 si ta CNN réduit largeur par 4

    alphabet = "abcdefghijklmnopqrstuvwxyz "
    char_to_num = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 = blank

    images = []
    labels = []
    input_lengths = []
    label_lengths = []

    with open(transcript_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    max_input_length = img_width // reduction_factor  # ex: 128//4=32

    for line in lines:
        img_name, transcript = line.strip().split("|")
        img_path = os.path.join(spectrogram_dir, img_name)
        if not os.path.exists(img_path):
            continue

        label_seq = [char_to_num.get(c, 0) for c in transcript if c in char_to_num]
        if len(label_seq) == 0:
            continue

        if len(label_seq) > max_input_length:
            continue  # transcription trop longue

        img = Image.open(img_path).convert('L')
        img = img.resize((img_width, img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)

        images.append(img_array)
        labels.append(label_seq)
        input_lengths.append(max_input_length)  # même pour tous, correct maintenant
        label_lengths.append(len(label_seq))

    max_label_len = max(label_lengths)
    labels_padded = np.zeros((len(labels), max_label_len), dtype=np.int32)
    for i, label_seq in enumerate(labels):
        labels_padded[i, :len(label_seq)] = label_seq

    images = np.array(images, dtype=np.float32)
    input_lengths = np.array(input_lengths, dtype=np.int32)
    label_lengths = np.array(label_lengths, dtype=np.int32)

    return images, labels_padded, input_lengths, label_lengths




if __name__ == "__main__":
    prepare_dataset()
