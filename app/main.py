from fastapi import FastAPI, File, UploadFile
from app.preprocess import preprocess_image
from app.model import predict_image

app = FastAPI(title="Cats vs Dogs Prediction API")

@app.get("/")
def home():
    return {"message": "Cats vs Dogs API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 1. Preprocess image bytes
    image_array = preprocess_image(contents)
    
    # 2. Get prediction (returns a numpy array like [[0.85]])
    prediction = predict_image(image_array)
    
    # 3. Extract the scalar float value from the numpy array
    # Adjust indexing if your model output shape is different
    probability = float(prediction[0][0]) 
    
    # 4. Determine label (assuming binary: > 0.5 is Dog, else Cat)
    label = "Dog" if probability > 0.5 else "Cat"
    
    return {
        "filename": file.filename,
        "prediction": label,
        "raw_prediction": round(probability, 6)
    }