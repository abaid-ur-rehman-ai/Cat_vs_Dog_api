from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "Cat_vs_Dog .keras"

# Load ONCE, when this module is first imported — not per request
model = load_model(str(MODEL_PATH))

def predict_image(image_array):
    prediction = model.predict(image_array)
    return prediction