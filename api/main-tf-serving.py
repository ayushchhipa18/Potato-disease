from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# Allow cross-origin requests from frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/potato-disease/model/potatoes_model/1/1.keras:predict"

# Class names used for prediction result
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/")
async def root():
    return {"message": "Welcome to Potato Disease Classifier API"}


@app.get("/favicon.ico")
async def favicon():
    return {}


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize if your model expects fixed size
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("🔥 /predict endpoint hit, file received:", file.filename)

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    json_data = {"instances": img_batch.tolist()}

    try:
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()
        prediction = np.array(response.json()["predictions"][0])
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {"class": predicted_class, "confidence": confidence}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # uvicorn.run("your_script_name:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("main-tf-serving:app", host="127.0.0.1", port=8000, reload=True)
