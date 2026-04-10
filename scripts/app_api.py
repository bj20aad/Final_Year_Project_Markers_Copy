from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import joblib
import numpy as np
import uvicorn

app = FastAPI()
model_session = None
scaler = None

class Packet(BaseModel):
    features: list

@app.on_event("startup")
def load_model():
    global model_session, scaler
    model_session = ort.InferenceSession('output/vae_model.onnx')
    scaler = joblib.load('output/scaler_vae.save')
    print("Model Loaded!")

@app.post("/predict")
def predict(packet: Packet):
    data = np.array([packet.features])
    scaled = scaler.transform(data).astype(np.float32)
    input_name = model_session.get_inputs()[0].name
    recon = model_session.run(None, {input_name: scaled})[0]
    mse = float(np.mean(np.power(scaled - recon, 2)))
    is_anomaly = mse > 0.05
    return {"anomaly_score": mse, "is_attack": is_anomaly}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)