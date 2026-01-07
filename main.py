from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import csv
import time
import random
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- CONFIG ---
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
MODEL_VERSION = "v1.0"

app = FastAPI()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock Database
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("password123"), 
        "role": "admin"
    }
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "prediction_logs.csv"
FLAGGED_FILE = "flagged_jobs.csv"

# --- HELPER: INITIALIZE CSV FILES ---
def init_csv():
    configs = [
        # FIXED: 4 Columns for History
        (LOG_FILE, ["timestamp", "description_length", "prediction", "confidence"]),
        # FIXED: 5 Columns for Audit Trail (as per your screenshot)
        (FLAGGED_FILE, ["timestamp", "reason", "predicted", "snippet", "comments"])
    ]
    for file, cols in configs:
        if not os.path.exists(file) or os.stat(file).st_size == 0:
            with open(file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(cols)

init_csv()

# Load ML Models
try:
    model = joblib.load('fake_job_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ SecureGuard Brain Loaded!")
except Exception as e:
    print(f"⚠️ Warning: Model files missing! Error: {e}")

class JobInput(BaseModel):
    description: str

class FeedbackInput(BaseModel):
    description: str
    prediction: str
    reason: str

# --- AUTH FUNCTIONS ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401)
        user = users_db.get(username)
        return user
    except JWTError:
        raise HTTPException(status_code=401)

# --- USER ROUTES ---

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid Credentials")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
def predict_job(data: JobInput):
    text = data.description.strip()
    
    if len(text) < 15:
        return {"result": "Incomplete Data", "confidence_score": 0, "detail": "Min 15 chars"}
    
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    result = "Fake" if pred == 1 else "Real"
    conf = round(float(max(prob) * 100), 2)
    
    # FIXED: Writing 4 values to match Prediction History headers
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(text), result, conf])
    
    return {"result": result, "confidence_score": conf}

@app.post("/feedback")
def save_feedback(fb: FeedbackInput):
    try:
        # FIXED: Writing 5 values to match Audit Trail headers
        # Columns: timestamp, reason, predicted, snippet, comments
        with open(FLAGGED_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                fb.reason, 
                fb.prediction, 
                fb.description[:100].replace('\n', ' '), # snippet
                "User Flagged" # comments
            ])
            f.flush()
        return {"message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ADMIN ROUTES ---

@app.get("/admin/stats")
def get_admin_stats(current_user: dict = Depends(get_current_user)):
    try:
        df = pd.read_csv(LOG_FILE)
        return {
            "total": len(df),
            "fake": int((df['prediction'] == 'Fake').sum()),
            "real": int((df['prediction'] == 'Real').sum()),
            "model_version": MODEL_VERSION
        }
    except:
        return {"total": 0, "fake": 0, "real": 0, "model_version": MODEL_VERSION}

@app.get("/admin/flagged-jobs")
def get_flags(current_user: dict = Depends(get_current_user)):
    if os.path.exists(FLAGGED_FILE):
        try:
            df = pd.read_csv(FLAGGED_FILE).fillna("N/A")
            return df.iloc[::-1].to_dict(orient="records")
        except:
            return []
    return []

@app.post("/admin/retrain")
def retrain(current_user: dict = Depends(get_current_user)):
    global MODEL_VERSION
    time.sleep(2) 
    MODEL_VERSION = f"v1.{random.randint(1,9)}"
    return {"message": "Success"}

@app.get("/admin/export")
def export_audit(current_user: dict = Depends(get_current_user)):
    if not os.path.exists(FLAGGED_FILE):
        raise HTTPException(status_code=404, detail="No audit logs")
    return FileResponse(path=FLAGGED_FILE, filename="Audit_Trail.csv", media_type='text/csv')

@app.get("/admin/export-predictions")
def export_all_logs(current_user: dict = Depends(get_current_user)):
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No history logs")
    return FileResponse(path=LOG_FILE, filename="Full_History.csv", media_type='text/csv')