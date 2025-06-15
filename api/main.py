from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

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

# ✅ Load the model from your .keras or .h5 path
MODEL = tf.keras.models.load_model("/home/ayush/ishu/potato-disease/saved_models/1.keras")
# Or use: MODEL = tf.keras.models.load_model("/home/ayush/ishu/potato-disease/potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # change according to your classes


@app.get("/")
def read_root():
    return {"message": "Welcome to the Potato Disease API"}


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # resize if needed
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
