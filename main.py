from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import joblib
import pandas as pd
import os
import csv
import random
import time

# --- CONFIGURATION ---
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MODEL_VERSION = "v1.0"

app = FastAPI(title="Fake Job Prediction System (Production Ready)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (Python 3.13 Safe)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simulated DB
hashed_password = pwd_context.hash("password123")
users_db = {"admin": {"username": "admin", "hashed_password": hashed_password, "role": "admin"}}

# --- FILES SETUP ---
LOG_FILE = "prediction_logs.csv"
FLAGGED_FILE = "flagged_jobs.csv"

# Initialize Logs if not exist
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["timestamp", "description_length", "prediction", "confidence"])
    df.to_csv(LOG_FILE, index=False)

# --- MODELS ---
try:
    model = joblib.load('fake_job_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Models loaded successfully!")
except:
    print("⚠️ Models not found.")

class JobInput(BaseModel):
    description: str

class FeedbackInput(BaseModel):
    description: str
    prediction: str
    reason: str
    comments: str = None

# --- AUTH FUNCTIONS ---
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None: raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    return users_db.get(payload.get("sub"))

# --- ROUTES ---

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": create_token({"sub": user["username"], "role": user["role"]}), "token_type": "bearer"}

@app.post("/predict")
def predict_job(data: JobInput):
    if len(data.description) < 10: raise HTTPException(status_code=400, detail="Text too short")
    
    # Prediction
    vec = vectorizer.transform([data.description])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    result = "Fake" if pred == 1 else "Real"
    conf = float(max(prob) * 100)
    
    # Log Activity (For Dashboard Stats)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), len(data.description), result, conf])
    
    return {"result": result, "confidence_score": round(conf, 2)}

@app.post("/feedback")
def save_feedback(fb: FeedbackInput):
    file_exists = os.path.exists(FLAGGED_FILE)
    with open(FLAGGED_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["timestamp", "reason", "predicted", "snippet", "comments"])
        writer.writerow([datetime.now().isoformat(), fb.reason, fb.prediction, fb.description[:50].replace('\n',' '), fb.comments])
    return {"message": "Saved"}

# --- ADMIN DASHBOARD APIs ---

@app.get("/admin/stats")
def get_stats(user: dict = Depends(get_current_user)):
    if user["role"] != "admin": raise HTTPException(status_code=403)
    
    total_preds = 0
    fake_count = 0
    real_count = 0
    
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            total_preds = len(df)
            counts = df['prediction'].value_counts()
            fake_count = int(counts.get('Fake', 0))
            real_count = int(counts.get('Real', 0))
        except: pass
        
    return {
        "total": total_preds,
        "fake": fake_count,
        "real": real_count,
        "model_version": MODEL_VERSION
    }

@app.get("/admin/flagged-jobs")
def get_flags(user: dict = Depends(get_current_user)):
    if user["role"] != "admin": raise HTTPException(status_code=403)
    if os.path.exists(FLAGGED_FILE):
        return pd.read_csv(FLAGGED_FILE).to_dict(orient="records")
    return []

@app.get("/admin/export")
def export_data(user: dict = Depends(get_current_user)):
    if user["role"] != "admin": raise HTTPException(status_code=403)
    if os.path.exists(FLAGGED_FILE):
        return FileResponse(FLAGGED_FILE, filename=f"flagged_report_{datetime.now().date()}.csv")
    raise HTTPException(status_code=404, detail="No data to export")

@app.post("/admin/retrain")
def retrain_model(user: dict = Depends(get_current_user)):
    if user["role"] != "admin": raise HTTPException(status_code=403)
    # Simulating Retraining Process
    time.sleep(3) # Fake processing time
    global MODEL_VERSION
    MODEL_VERSION = f"v1.{random.randint(1,9)}"
    return {"message": "Model Retrained Successfully", "new_version": MODEL_VERSION}