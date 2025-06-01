import numpy as np
from PIL import Image

# ====================== DATA LOADING ======================
def load_spectrogram_image(path):
    """Load and preprocess spectrogram image"""
    img = Image.open(path).convert("L")
    img = img.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)  # (128, 128, 1)

def get_training_sample():
    """Generate a training sample with proper shapes"""
    X = load_spectrogram_image("static/spectrogramme.png")  # (128, 128, 1)
    X = np.expand_dims(X, axis=0)  # Add batch dim (1, 128, 128, 1)
    
    # Example encoding for "bonjour" (adapt to your character set)
    y_text = "bonjour"
    char_to_int = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    y = np.array([[char_to_int[c] for c in y_text]], dtype=np.int32)  # (1, 7)
    
    # CTC requirements
    input_length = np.array([[32]], dtype=np.int32)  # Timesteps after CNN
    label_length = np.array([[len(y_text)]], dtype=np.int32)
    
    return X, y, input_length, label_length