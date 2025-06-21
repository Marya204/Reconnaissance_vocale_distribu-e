import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
DATASETS = {
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
}
BASE_DIR = "datasets"
INPUT_DIR = os.path.join(BASE_DIR, "LibriSpeech/train-clean-100")
OUTPUT_DIR = "data/spectrograms"
TRANSCRIPT_FILE = "data/transcriptions.txt"
MAX_FILES = None  # No limit, process all files in train-clean-100
MAX_DURATION_SEC = 30
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
alphabet_set = set(ALPHABET)

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

# --- Sauvegarde du spectrogramme ---
def save_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y)
    duration = len(y) / sr
    if duration > MAX_DURATION_SEC:
        return False
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=32)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = (mel_norm * 255).astype(np.uint8)
    img = Image.fromarray(mel_img)
    img.save(output_path)
    return True

# --- Préparation du dataset ---
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

    transcripts = []
    count = 0
    print("Génération des spectrogrammes...")
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if MAX_FILES is not None and count >= MAX_FILES:
                break
            if not file.endswith(".flac"):
                continue
            full_path = os.path.join(root, file)
            base_name = Path(file).stem
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
            # Load and process original audio
            try:
                y, sr = librosa.load(full_path, sr=16000)
                y, _ = librosa.effects.trim(y)
            except Exception as e:
                print(f"[LOAD ERROR] {base_name}: {e}")
                continue
            # Generate and save original spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=32)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            mel_img = (mel_norm * 255).astype(np.uint8)
            img = Image.fromarray(mel_img)
            img.save(out_path)
            # Process label
            transcription_file = os.path.join(root, file).replace(".flac", ".txt").rsplit("-", 1)[0] + ".trans.txt"
            if not os.path.exists(transcription_file):
                continue
            with open(transcription_file, "r", encoding="utf-8") as tf:
                for line in tf:
                    if line.startswith(base_name):
                        _, text = line.strip().split(" ", 1)
                        label = text.lower().replace("'", "").strip()
                        label = re.sub(r"[^a-z ]", "", label)
                        label = ' '.join(label.split())
                        word_count = len(label.split())
                        if word_count < 1 or word_count > 10:
                            break
                        if set(label) - alphabet_set:
                            print(f"[⚠️ Caractères invalides] {base_name} → {label}")
                            break
                        transcripts.append(f"{base_name}.png|{label}")
                        count += 1
                        break
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        for line in transcripts:
            f.write(line + "\n")
    print(f"\n✅ {count} fichiers audio traités avec transcriptions valides.")
    print(f"📄 Fichier généré : {TRANSCRIPT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
