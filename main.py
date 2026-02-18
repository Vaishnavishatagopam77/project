from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import random
from datetime import timedelta, datetime
import secrets
from pydantic import BaseModel, ConfigDict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Local imports
from content_db import CONTENT, TOPICS
from feynman import feynman_explain_batch
from quiz_generator import generate_reassessment_quiz
import models
from database import engine, get_db
import auth_handler

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class EvaluateQuizRequest(BaseModel):
    topic: str
    user_answers: Dict[str, str]

class EvaluateQuizResponse(BaseModel):
    score: float
    passed: bool
    weak_concepts: List[str]
    results: List[Dict[str, Any]]

class ReassessmentRequest(BaseModel):
    topic: str
    previous_questions: List[str]

class FeynmanRequest(BaseModel):
    topic: str
    concepts: List[str] = []

class QuizResultSchema(BaseModel):
    id: int
    topic: str
    score: float
    passed: bool
    timestamp: str
    model_config = ConfigDict(from_attributes=True)

# --- Auth Endpoints ---

@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth_handler.get_password_hash(user.password)
    db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token = auth_handler.create_access_token(
        data={"sub": user.username}, expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth_handler.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_handler.create_access_token(
        data={"sub": user.username}, expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == req.email).first()
    # Always return success message to avoid revealing existence
    if not user:
        return {"message": "If the email exists, a reset link has been sent."}

    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    pr = models.PasswordReset(user_id=user.id, token=token, expires_at=expires_at)
    db.add(pr)
    db.commit()

    reset_link = f"https://example.com/reset-password?token={token}"
    # In production, send email. For now print to server logs.
    print(f"Password reset link for {user.email}: {reset_link}")

    return {"message": "If the email exists, a reset link has been sent."}


@app.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    pr = db.query(models.PasswordReset).filter(models.PasswordReset.token == req.token).first()
    if not pr or pr.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user = db.query(models.User).filter(models.User.id == pr.user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")

    user.hashed_password = auth_handler.get_password_hash(req.new_password)
    db.delete(pr)
    db.commit()

    return {"message": "Password has been reset successfully."}

@app.get("/users/me")
def read_users_me(current_user: models.User = Depends(auth_handler.get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "id": current_user.id}

# --- Content Endpoints ---

@app.get("/topics")
def get_topics():
    return TOPICS

@app.get("/content/{topic_code}")
def get_content(topic_code: str):
    if topic_code not in CONTENT:
        raise HTTPException(status_code=404, detail="Topic not found")
    data = CONTENT[topic_code]
    return {
        "topic": TOPICS.get(topic_code),
        "context": data["context"],
        "questions": data["questions"]
    }

@app.post("/evaluate")
def evaluate_quiz(
    req: EvaluateQuizRequest, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth_handler.get_current_user)
):
    if req.topic not in CONTENT:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    data = CONTENT[req.topic]
    questions = data["questions"]
    
    correct_count = 0
    weak_concepts = []
    results = []
    
    for q in questions:
        qid = str(q["id"])
        user_ans = req.user_answers.get(qid)
        is_correct = (user_ans == q["a"])
        if is_correct:
            correct_count += 1
        else:
            weak_concepts.append(q["c"])
        results.append({
            "id": q["id"],
            "q": q["q"],
            "user_ans": user_ans,
            "correct_ans": q["a"],
            "is_correct": is_correct,
            "explanation": q["e"]
        })
        
    score = (correct_count / len(questions)) * 100
    passed = score >= 70
    
    # Save to DB
    db_result = models.QuizResult(
        user_id=current_user.id,
        topic=req.topic,
        score=score,
        passed=passed,
        weak_concepts=list(set(weak_concepts))
    )
    db.add(db_result)
    db.commit()
    
    return EvaluateQuizResponse(
        score=score,
        passed=passed,
        weak_concepts=list(set(weak_concepts)),
        results=results
    )

@app.get("/dashboard/stats")
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth_handler.get_current_user)
):
    # Get all results for user
    results = db.query(models.QuizResult).filter(models.QuizResult.user_id == current_user.id).all()
    
    total_quizzes = len(results)
    if total_quizzes == 0:
        return {"average_score": 0, "best_score": 0, "total_quizzes": 0, "history": []}
    
    scores = [r.score for r in results]
    avg_score = sum(scores) / total_quizzes
    best_score = max(scores)
    
    # History for graph (last 10 attempts)
    history = [
        {"attempt": i+1, "score": r.score, "topic": r.topic, "date": r.timestamp} 
        for i, r in enumerate(results[-10:])
    ]
    
    return {
        "average_score": avg_score,
        "best_score": best_score,
        "total_quizzes": total_quizzes,
        "history": history
    }

@app.post("/feynman_explain")
def feynman_explain_endpoint(req: FeynmanRequest):
    if req.topic not in CONTENT:
        raise HTTPException(status_code=404, detail="Topic not found")
    context = CONTENT[req.topic]["context"]
    if not req.concepts:
        static_feynman = CONTENT[req.topic].get("feynman", {})
        concepts_to_explain = list(static_feynman.keys())
    else:
        concepts_to_explain = req.concepts
    return feynman_explain_batch(concepts_to_explain, context)

@app.post("/reassessment")
def get_reassessment(req: ReassessmentRequest):
    if req.topic not in CONTENT:
        raise HTTPException(status_code=404, detail="Topic not found")
    context = CONTENT[req.topic]["context"]
    generated_questions = generate_reassessment_quiz(context, req.previous_questions, num_questions=5)
    if not generated_questions:
        generated_questions = []
        static_re = CONTENT[req.topic].get("reassessment", {})
        for cat_qs in static_re.values():
            generated_questions.extend(cat_qs)
        if len(generated_questions) > 5:
            generated_questions = random.sample(generated_questions, 5)
    return generated_questions

# --- Static Files & SPA Serve ---

import os

# Define the path to the frontend build directory
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")

# Mount assets directory (Vite puts assets in dist/assets)
if os.path.exists(os.path.join(frontend_dist, "assets")):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
# Catch-all route to serve React
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Check if the file exists in the root of dist (e.g. favicon.ico, manifest.json)
    file_path = os.path.join(frontend_dist, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fallback to index.html for SPA routing
    index_path = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not built. Please run `npm run build` in /frontend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
