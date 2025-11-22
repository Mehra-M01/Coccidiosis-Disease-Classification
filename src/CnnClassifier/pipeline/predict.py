import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -----------------------------
# Load the model ONLY ONCE here
# -----------------------------
MODEL_PATH = os.path.join("artifacts", "training", "model.h5")
print("Loading model... (only once at startup)")
model = load_model(MODEL_PATH, compile = False)
print("Model loaded successfully!")


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Model already loaded globally, no need to load again

        img = image.load_img(self.filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        result = np.argmax(preds)

        if result == 1:
            prediction = "Healthy"
        else:
            prediction = "Coccidiosis"

        return [{"image": prediction}]
