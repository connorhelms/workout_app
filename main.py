from flask import Flask, request, render_template_string, flash, redirect, url_for, render_template, send_from_directory, jsonify, session
import openai
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, current_user, logout_user, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
import base64
from moviepy.editor import VideoFileClip
import cv2
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create tables
with app.app_context():
    db.create_all()

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('home'))
    
    file = request.files['file']
    exercise_type = request.form.get('exercise_type', '')
    notes = request.form.get('notes', '')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video and get analysis
        result = process_video(filepath, exercise_type, notes)
        
        # Create new workout session
        workout_session = WorkoutSession(
            user_id=current_user.id,
            file_type='video',
            video_filename=filename,
            exercise=result['exercise'],
            analysis=result['analysis'],
            feedback=result['feedback'],
            notes=notes,
            character_id=result['character_id'],
            character_name=result['character_name']
        )
        
        db.session.add(workout_session)
        db.session.commit()
        
        return redirect(url_for('session_detail', session_id=workout_session.id))
    
    flash('Invalid file type')
    return redirect(url_for('home'))

# Update WorkoutSession model (remove video_url)
class WorkoutSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    video_filename = db.Column(db.String(300), nullable=False)
    exercise = db.Column(db.String(50))
    analysis = db.Column(db.Text)
    feedback = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    character_id = db.Column(db.String(50))
    character_name = db.Column(db.String(100))

# Rest of your existing code...

# Make sure this is at the bottom of main.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 