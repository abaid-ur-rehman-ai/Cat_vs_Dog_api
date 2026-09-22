# Dogs vs Cats Classification using Transfer Learning (MobileNetV2)

This project classifies images of dogs and cats using **Transfer Learning** with the **MobileNetV3** model in Keras. The model is also deployed as an API.

## Live Demo
https://cat-vs-dog-api.onrender.com/docs


## Project Overview

- Dataset: Dogs vs Cats
- Model: MobileNet (Pre-trained on ImageNet)
- Framework: TensorFlow / Keras
- Type: Image Classification (Binary)
- Deployment: FastAPI (Live API)

## Features

- Transfer Learning with MobileNet
- Image preprocessing and data augmentation
- Model training and evaluation
- Real-time prediction using API
- Swagger UI documentation

## Tech Stack

- Python
- TensorFlow / Keras
- FastAPI
- Uvicorn
- NumPy
- MobileNet

## Model Performance

- Training Accuracy: (0.9916)
- Validation Accuracy: (0.9836)

## How to Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
