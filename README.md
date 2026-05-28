# Spam Email Detector API

A FastAPI-based REST API for detecting spam emails using a trained machine learning model.

## Overview

This API provides a `/predict` endpoint that analyzes email content and returns whether it's spam along with a confidence score.

## Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Model**: Pickled ML model (email spam detection)
- **Validation**: Pydantic

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd api
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API

```bash
python main.py
```

The server will start on `http://localhost:5000`.

### API Endpoints

#### Root
- **GET** `/`
- Returns a welcome message

#### Predict Spam
- **POST** `/predict`
- **Body**: `{"email": "email content string"}`
- **Response**:
  ```json
  {
    "is_spam": true,
    "confidence": 0.95
  }
  ```

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "Congratulations! You won a prize. Click here to claim."}'
```

## Project Structure

```
api/
├── src/
│   ├── app.py                  # FastAPI application
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── assets/
│       ├── vectorizer.pkl      # TF-IDF vectorizer
│       └── EmailSpamDetectionModel.pkl  # Trained model
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## License

Apache 2.0 License
