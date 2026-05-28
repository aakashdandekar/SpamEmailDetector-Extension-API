import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.models.schemas import Request

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credintials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent

vectorizer_path = BASE_DIR / "assets" / "vectorizer.pkl"
model_path = BASE_DIR / "assets" / "EmailSpamDetectionModel.pkl"

with open(vectorizer_path, "rb") as f:
    vectorizer = joblib.load(f)

with open(model_path, "rb") as f:
    model = joblib.load(f)

@app.post('/predict')
async def predict(
    request: Request
):
    try:
        text = vectorizer.transform([request.email])

        prediction = model.predict(text)[0]
        confidence = float(model.predict_proba(text)[0][1])

        return {
            "is_spam": bool(prediction == 1),
            "confidence": round(confidence, 5)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    
@app.get('/')
async def root():
    return {
        "message": "Email Spam Detetion"
    }
