import sqlite3
import hashlib
import os
from datetime import datetime
import re

DB_PATH = "neovihar_users.db"

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # User progress table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic_code TEXT NOT NULL,
            progress_percentage INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            quiz_attempts INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, topic_code)
        )
    """)
    
    # Password reset tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            reset_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    return True, "Valid"

def register_user(username, email, password):
    """Register a new user"""
    try:
        if not validate_email(email):
            return False, "Invalid email format"
        
        is_valid, msg = validate_password(password)
        if not is_valid:
            return False, msg
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        
        user_id = cursor.lastrowid
        
        # Create default progress entries for standard AI topics
        topics = ["AI", "ML", "DL", "DS", "CV"]
        for topic_code in topics:
            cursor.execute(
                "INSERT INTO user_progress (user_id, topic_code) VALUES (?, ?)",
                (user_id, topic_code)
            )
        
        conn.commit()
        conn.close()
        
        return True, "Registration successful! Please login."
    
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        elif "email" in str(e):
            return False, "Email already registered"
        return False, str(e)
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user login"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute(
            "SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {"id": user[0], "username": user[1], "email": user[2]}
        else:
            return False, "Invalid username or password"
    
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

def get_user_progress(user_id):
    """Get user's learning progress for all topics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT topic_code, progress_percentage, score, quiz_attempts FROM user_progress WHERE user_id = ?",
            (user_id,)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        progress_dict = {}
        for topic_code, progress, score, attempts in results:
            progress_dict[topic_code] = {
                "progress": progress,
                "score": score,
                "attempts": attempts
            }
        
        return progress_dict
    
    except Exception as e:
        return {}

def update_user_progress(user_id, topic_code, progress_percentage, score):
    """Update user's progress for a specific topic"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """UPDATE user_progress 
               SET progress_percentage = ?, score = ?, quiz_attempts = quiz_attempts + 1, last_updated = CURRENT_TIMESTAMP
               WHERE user_id = ? AND topic_code = ?""",
            (progress_percentage, score, user_id, topic_code)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        return False

def request_password_reset(email):
    """Generate password reset token"""
    try:
        import secrets
        from datetime import timedelta
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Email not found"
        
        user_id = user[0]
        reset_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=1)
        
        cursor.execute(
            "INSERT INTO password_resets (user_id, reset_token, expires_at) VALUES (?, ?, ?)",
            (user_id, reset_token, expires_at)
        )
        
        conn.commit()
        conn.close()
        
        return True, reset_token
    
    except Exception as e:
        return False, str(e)

def reset_password(email, new_password):
    """Reset password for a user"""
    try:
        is_valid, msg = validate_password(new_password)
        if not is_valid:
            return False, msg
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(new_password)
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (password_hash, email)
        )
        
        # Remove all reset tokens for this user
        cursor.execute("DELETE FROM password_resets WHERE user_id = (SELECT id FROM users WHERE email = ?)", (email,))
        
        conn.commit()
        conn.close()
        
        return True, "Password reset successful!"
    
    except Exception as e:
        return False, str(e)

def user_exists(username):
    """Check if username exists"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        return user is not None
    except:
        return False

def email_exists(email):
    """Check if email exists"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        return user is not None
    except:
        return False
