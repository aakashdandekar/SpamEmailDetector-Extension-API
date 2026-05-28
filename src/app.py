import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.models.schemas import Request

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credintials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("./assets/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("./assets/EmailSpamDetectionModel.pkl", "rb") as file:
    model = pickle.load(file)

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