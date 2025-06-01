import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .utils import get_training_sample
from .asr_model import build_model


# Replace this with your actual data loading logic
X_train, y_train, input_length_train, label_length_train = get_training_sample()

# Build the model
train_model, pred_model = build_model(input_dim=(128, 128, 1), output_dim=29)

# Print model summary
print(train_model.summary())

# Train the model
train_model.fit(
    x=[X_train, y_train, input_length_train, label_length_train],
    y={'ctc_loss': tf.zeros_like(y_train)},  # Dummy target since we use the loss function directly
    batch_size=32,
    epochs=10
)