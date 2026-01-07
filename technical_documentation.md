# ⚙️ Technical Documentation: SecureGuard Fraud Detection Engine

## 1. System Architecture & Design
The SecureGuard system is built on a **Stateless Distributed Architecture**, ensuring high performance and clear separation between logic and presentation.

* **Frontend (Presentation Layer)**: A modern, responsive Neon-themed UI built with HTML5, CSS3, and JavaScript. It communicates with the backend via Asynchronous REST API calls.
* **Security Layer (Authentication)**: Implements OAuth2 with JWT (JSON Web Tokens). This layer intercepts every administrative request to validate the cryptographic signature.
* **Backend (Logic Layer)**: Powered by FastAPI. It handles request routing, business logic, and integrates the Machine Learning models.
* **Intelligence Layer (ML)**: Utilizes a Random Forest Classifier and TF-IDF vectorization to analyze job text patterns for fraud detection.



## 2. API Documentation
All API responses are returned in JSON format. Admin endpoints require a `Bearer <token>` in the Authorization header.

### 2.1 Prediction Endpoints
* **POST `/predict`**
    * **Description**: Analyzes job text for fraud patterns.
    * **Payload**: `{ "description": "Job description string" }`
    * **Response**: `{ "result": "Fake/Real", "confidence_score": 92.5 }`

### 2.2 Feedback & Logging
* **POST `/feedback`**
    * **Description**: Allows users to report false positives/negatives.
    * **Payload**: `{ "description": "string", "prediction": "string", "reason": "string" }`

### 2.3 Administrative Endpoints (Secure)
* **GET `/admin/stats`**: Returns a summary of total scans, fake vs real counts for charts.
* **POST `/admin/retrain`**: Triggers the pipeline to update the model version.
* **GET `/admin/export-predictions`**: Streams the full `prediction_logs.csv`.
* **GET `/admin/export`**: Streams the `flagged_jobs.csv` (Audit Trail).

## 3. Database Schema (Persistence Layer)
The system uses a **High-Performance Flat-File Schema (CSV)** for data persistence, ensuring zero-database overhead and easy portability.

### 3.1 Table: `prediction_logs.csv`
| Column | Data Type | Description |
| :--- | :--- | :--- |
| timestamp | String | The exact date and time of the scan. |
| description_length | Integer | Length of the analyzed job post. |
| prediction | String | "Real" or "Fake". |
| confidence | Float | The probability score assigned by the model. |

### 3.2 Table: `flagged_jobs.csv`
| Column | Data Type | Description |
| :--- | :--- | :--- |
| timestamp | String | Date of the user report. |
| reason | String | User-provided reason for flagging. |
| predicted | String | The result the AI originally gave. |
| snippet | String | First 100 characters of the job text. |
| comments | String | System/Admin internal notes. |

## 4. Setup and Installation Guide
### 4.1 Prerequisites
* Python 3.10 or higher
* Package Manager: `pip`

### 4.2 Installation Steps
1.  **Clone Project**: Place all files in a project directory.
2.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn scikit-learn joblib python-jose[cryptography] passlib[bcrypt] pandas
    ```
3.  **Launch Server**:
    ```bash
    uvicorn main:app --reload --port 8000
    ```

## 5. Troubleshooting Guide
| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| **401 Unauthorized** | Expired JWT Token. | Re-login via `login.html`. |
| **CORS Policy Block** | Origin mismatch. | Add frontend URL to `allow_origins` in `main.py`. |
| **Empty Dashboard** | CSV files not found. | Run 1-2 predictions and flags to generate files. |
| **Data Mismatch** | Column count error. | Ensure `init_csv()` matches the `writerow` logic in `main.py`. |