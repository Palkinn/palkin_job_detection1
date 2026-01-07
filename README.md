### SECUREGUARD: FAKE JOB INTELLIGENCE SYSTEM
### 1. Project Overview
SecureGuard is a comprehensive Full-Stack Web Application designed to identify and mitigate fraudulent job postings using Machine Learning. The system features a FastAPI backend for high-speed model inference and a modern, responsive Neon-UI frontend for an interactive user experience.

### 2. Key Features
Real-time Prediction: Instantly analyzes job descriptions to detect linguistic patterns of fraud.
Confidence Scoring: Displays a granular percentage score indicating the AI's certainty.
Admin Intelligence Center: A secure dashboard with Chart.js visuals to monitor fraud distribution.
Feedback & Audit System: Users can flag incorrect results, which are saved with specific reasons into a multi-column Audit Trail.
JWT Security: Protected admin routes using JSON Web Tokens and encrypted password hashing.
### 3. Tech Stack
Frontend: HTML5, CSS3 (Neon Design), JavaScript (Fetch API), Chart.js.
Backend: Python 3.9+, FastAPI, Uvicorn.
ML Engine: Scikit-Learn (Random Forest Classifier), TF-IDF Vectorizer.
Persistence: Structured CSV Logging (History & Audit Trail).
### 4. Prerequisites
Python 3.9 or higher.

Pip (Python Package Manager).

Modern Browser (Chrome, Edge, or Firefox).

### 5. Installation & Setup
### Step 1: File Verification
Ensure the following core files are in your project directory:

main.py (Backend Logic)

fake_job_model.pkl & tfidf_vectorizer.pkl (AI Models)

index.html, login.html, admin.html (UI Layers)

### Step 2: Dependency Injection
Open your terminal in the project folder and run:

Bash

pip install fastapi uvicorn scikit-learn joblib pandas python-jose[cryptography] passlib[bcrypt]
### Step 3: Launching the System
Start the backend server using the following command:

Bash

uvicorn main:app --reload --port 8000
### 6. Administrative Configuration
Default Access: Admins can log in via login.html using the configured credentials in main.py.
Data Export: Perfectly aligned CSV files can be exported directly from the Admin Dashboard for analysis in Microsoft Excel.
Model Update: The "Re-calibrate AI" feature allows admins to increment the system version and refresh the prediction logic.
### 7. Troubleshooting
401 Error: Session expired. Re-login via login.html.
Empty Charts: Perform at least 5-10 predictions to populate the visual analytics.
CORS Issues: Ensure the frontend is accessing the correct local port (default: 8000).