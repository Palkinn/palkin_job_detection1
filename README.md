# Fake Job Prediction System (Milestone 3)

## Project Overview
This project is a Full-Stack Web Application that uses Machine Learning to detect Fake Job Descriptions. It consists of a **FastAPI Backend** for model inference and an **HTML/JS Frontend** for the user interface.

## Features
- **Real-time Prediction:** Analyzes job descriptions to detect if they are Real or Fake.
- **Confidence Score:** Displays the model's confidence percentage.
- **Feedback System:** Users can flag incorrect predictions, which are saved to a CSV file for future model improvement.
- **Interactive UI:** Modern, responsive interface with visual indicators.

## Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
- **Backend:** Python, FastAPI, Uvicorn
- **ML Model:** Scikit-Learn (Random Forest), TF-IDF Vectorizer

## Prerequisites
- Python 3.8 or higher
- Pip (Python Package Manager)

## Installation & Setup

1. **Clone/Download the project folder.**
   Ensure the following files are present:
   - `main.py`
   - `index.html`
   - `fake_job_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `requirements.txt`

2. **Install Dependencies**
   Open your terminal in the project folder and run:
   ```bash
   pip install -r requirements.txt