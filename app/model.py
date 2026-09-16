from pathlib import Path
from tensorflow.keras.models import load_model

# Gets the directory of model.py (which is inside 'app')
BASE_DIR = Path(__file__).resolve().parent

# Goes up one level from 'app/' to the root directory, then into 'model/'
MODEL_PATH = BASE_DIR.parent / "model" / "Cat_vs_Dog .keras"

def load_cat_dog_model():
    model = load_model(str(MODEL_PATH))
    return model

def predict_image(image_array):
    model = load_cat_dog_model()
    prediction = model.predict(image_array)
    return prediction