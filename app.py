from flask import Flask, request, render_template_string, flash, redirect, url_for, render_template, send_from_directory, jsonify, session
import openai
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, current_user, logout_user, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict
import base64
from moviepy.editor import VideoFileClip
from difflib import get_close_matches
import string
import cv2
import numpy as np
import re

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key securely via environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")  # Now loaded from .env

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY") or 'your_secret_key_here'  # Use value from .env

# Initialize the database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models for Users and Workout Sessions
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class WorkoutSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'video' or 'image'
    video_filename = db.Column(db.String(300), nullable=False)
    exercise = db.Column(db.String(50))
    analysis = db.Column(db.Text)
    feedback = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    character_id = db.Column(db.String(50))
    character_name = db.Column(db.String(100))

    @property
    def formatted_date(self):
        """Return the date in AM/PM format"""
        return self.created_at.strftime('%Y-%m-%d %I:%M %p')

    user = db.relationship('User', backref='sessions')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """
    Check if the uploaded file has one of the allowed extensions.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, exercise_type, notes=""):
    """
    Process video and generate analysis using OpenAI API.
    """
    try:
        exercise = normalize_exercise_name(exercise_type)
        
        prompt = f"""
        As a strict and detail-oriented fitness trainer analyzing a {exercise} workout, provide a thorough critique. The user has provided these notes: {notes}

        Analyze the workout and provide specific feedback in these exact sections. Each section should contain your actual observations and recommendations, not descriptions of what to look for:

        1. Form Assessment
           - Describe the actual posture and alignment observed
           - Detail the specific movement patterns seen
           - Evaluate the actual range of motion achieved
           - Comment on the observed tempo and control

        2. Technical Flaws
           - List the specific form deviations observed
           - Describe the exact incorrect movements seen
           - Detail any compensatory patterns noticed

        3. Critical Improvements
           - List the specific corrections needed, based on observations
           - Identify the most urgent safety issues seen
           - Detail any mobility limitations observed

        4. Safety Concerns
           - List specific injury risks based on observed form
           - Describe dangerous movement patterns seen
           - Detail any stability issues noticed

        5. Corrective Actions
           - Provide specific cues to address the observed issues
           - Recommend specific mobility exercises needed
           - Suggest appropriate regression exercises

        6. Advanced Recommendations
           - List specific form refinements for improvement
           - Suggest appropriate progression exercises
           - Provide advanced technique tips based on current form

        Format each section with a numbered heading followed by bullet points of actual observations and recommendations. Add a blank line between sections. Be direct and specific about what you observed.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly experienced and strict fitness trainer. Provide specific, detailed observations and recommendations based on the workout being analyzed. Format responses with clear numbering and dashes only, adding a blank line between numbered sections. Do not use asterisks, bold text, or other formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message['content']
            
            # Clean up the formatting
            analysis = analysis.replace('\n\n\n', '\n\n')  # Remove triple line breaks
            analysis = analysis.replace('*', '')           # Remove any asterisks
            analysis = analysis.replace('**', '')          # Remove any double asterisks
            analysis = analysis.replace('•', '-')          # Standardize bullet points
            
            # Ensure single blank line between numbered sections
            analysis = re.sub(r'(\d+\..*?)(\d+\.)', r'\1\n\n\2', analysis, flags=re.DOTALL)
            
            # Generate personalized feedback based on the analysis
            feedback = generate_feedback(exercise, analysis)
            
            # Get the current character
            character_id = session.get('character_id', 'trainer')
            character = CHARACTERS[character_id]
            
            return {
                'exercise': exercise,
                'analysis': analysis,
                'feedback': feedback,
                'character_name': character.name,
                'character_id': character_id  # Add character ID to include emoji
            }
            
        except Exception as api_error:
            print(f"OpenAI API Error: {api_error}")
            return {
                'exercise': exercise,
                'analysis': f"Error generating analysis: {str(api_error)}. Please try again.",
                'feedback': "Unable to provide feedback at this time.",
                'character_name': 'AI Coach',
                'character_id': 'trainer'
            }
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            'exercise': "unspecified exercise",
            'analysis': f"Unable to analyze video properly: {str(e)}",
            'feedback': "Error processing the workout video.",
            'character_name': 'Personal Trainer',
            'character_id': 'trainer'
        }

@dataclass
class Character:
    id: str
    name: str
    emoji: str
    description: str
    prompt_style: str

CHARACTERS: Dict[str, Character] = {
    'trainer': Character(
        'trainer',
        'Personal Trainer',
        '💪',
        'Professional and encouraging coach focused on proper form and technique.',
        'You are a professional personal trainer. Provide clear, encouraging feedback focused on proper form and technique.'
    ),
    'goggins': Character(
        'goggins',
        'David Goggins',
        '😤',
        'No excuses! Push harder than you think possible.',
        'You are David Goggins. Be intense, use tough love, and push the user to be their absolute best. Use some profanity and be brutally honest about their form.'
    ),
    'musashi': Character(
        'musashi',
        'Musashi Miyamoto',
        '⚔️',
        'Ancient wisdom meets physical discipline.',
        'You are Musashi Miyamoto. Provide feedback that connects physical training with spiritual growth and mental discipline. Speak in a wise, philosophical manner.'
    ),
    'drill': Character(
        'drill',
        'Drill Sergeant',
        '🎖️',
        'Drop and give me twenty! Military-style motivation.',
        'You are a drill sergeant. Be loud, demanding, and use military-style motivation. Address the user as "recruit" and be extremely strict about form.'
    ),
    'chief': Character(
        'chief',
        'Master Chief',
        '🎮',
        'Spartan-level training and efficiency.',
        'You are Master Chief. Provide tactical, efficient feedback focused on maximum performance. Reference Spartan training and military precision.'
    ),
    'iroh': Character(
        'iroh',
        'Uncle Iroh',
        '🍵',
        'Wise guidance through the path of improvement.',
        'You are Uncle Iroh from Avatar: The Last Airbender. Provide wise, caring feedback that connects physical training with inner peace and balance. Use tea metaphors.'
    ),
    'durden': Character(
        'durden',
        'Tyler Durden',
        '👊',
        'Break free from your limitations.',
        'You are Tyler Durden. Be provocative and philosophical about physical improvement. Challenge societal norms while providing feedback about form.'
    )
}

def generate_feedback(exercise, analysis):
    character_id = session.get('character_id', 'trainer')
    character = CHARACTERS[character_id]
    
    prompt = f"""
    {character.prompt_style}
    
    A user performed a {exercise}. The analysis of their form indicates the following issues:
    {analysis}
    
    Provide feedback in character, addressing these issues and offering suggestions to improve the form.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": character.prompt_style},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        feedback = response['choices'][0]['message']['content']
    except Exception as e:
        feedback = f"Error generating feedback: {str(e)}"
    return feedback

@app.route('/', methods=['GET', 'POST'])
def home():
    current_hour = datetime.now().hour
    
    if 0 <= current_hour < 11:
        greeting = "Good morning"
    elif 11 <= current_hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    
    if current_user.is_authenticated:
        greeting += f", {current_user.username}"
        
    return render_template('index.html', greeting=greeting)

@app.route('/dashboard')
@login_required
def dashboard():
    sessions = WorkoutSession.query.filter_by(user_id=current_user.id).order_by(WorkoutSession.created_at.desc()).all()
    # Add title case formatting for exercise names
    for session in sessions:
        if session.exercise:
            # Convert exercise name to title case, preserving hyphens
            words = session.exercise.split()
            session.display_name = ' '.join(word.title() for word in words).replace('- ', '-')
    return render_template('dashboard.html', sessions=sessions)

@app.route('/session/<int:session_id>')
@login_required
def session_detail(session_id):
    session_record = WorkoutSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session_record:
        return "Session not found or access denied", 404
    return render_template('session_detail.html', session=session_record)

# Add this list of common exercise names
COMMON_EXERCISES = {
    'push ups', 'push-ups', 'pushups',
    'pull ups', 'pull-ups', 'pullups',
    'squats', 'squat',
    'deadlift', 'deadlifts',
    'bench press',
    'shoulder press',
    'lunges', 'lunge',
    'plank',
    'burpees',
    'jumping jacks',
    'sit ups', 'sit-ups', 'situps',
    'crunches',
    'bicep curls',
    'tricep extensions',
    'rows',
    'dips',
    'overhead press',
    'mountain climbers',
    'russian twists',
    'leg raises',
    'wall sits',
    'chin ups', 'chin-ups', 'chinups'
}

def normalize_exercise_name(exercise_input):
    """
    Normalize exercise name by:
    1. Converting to lowercase
    2. Finding closest match in common exercises
    3. Applying proper capitalization
    """
    if not exercise_input:
        return ""
        
    # Convert to lowercase for matching
    exercise_lower = exercise_input.lower().strip()
    
    # Find closest match in common exercises
    matches = get_close_matches(exercise_lower, COMMON_EXERCISES, n=1, cutoff=0.8)
    
    if matches:
        exercise_normalized = matches[0]
    else:
        exercise_normalized = exercise_lower
        
    # Capitalize words properly
    words = exercise_normalized.split()
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)

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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(request.url)
    
    video = request.files['video']
    
    if video.filename == '':
        flash('No video selected')
        return redirect(request.url)
    
    if video and allowed_file(video.filename):
        # Create a temporary file to check video duration
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + secure_filename(video.filename))
        video.save(temp_path)
        
        try:
            clip = VideoFileClip(temp_path)
            duration = clip.duration
            clip.close()
            
            if duration > 30:
                os.remove(temp_path)
                flash('Video must be 30 seconds or shorter')
                return redirect(request.url)
                
            # If duration is okay, proceed with your existing upload logic
            os.rename(temp_path, os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video.filename)))
            
            # Determine file type
            extension = video.filename.rsplit('.', 1)[1].lower()
            file_type = 'video' if extension in {'mp4', 'mov', 'avi', 'mkv'} else 'image'
            
            # Create workout session
            workout = WorkoutSession(
                user_id=current_user.id,
                video_filename=video.filename,
                file_type=file_type,
                notes=request.form.get('notes', ''),
                character_id=session.get('character_id', 'trainer'),
                character_name=session.get('character_name', 'Personal Trainer')
            )
            
            # Generate analysis based on file type
            if file_type == 'video':
                analysis = analyze_workout_video(os.path.join(app.config['UPLOAD_FOLDER'], video.filename), request.form.get('notes', ''))
            else:
                analysis = analyze_workout_image(os.path.join(app.config['UPLOAD_FOLDER'], video.filename), request.form.get('notes', ''))
            
            workout.exercise = analysis.get('exercise', 'Unknown Exercise')
            workout.analysis = analysis.get('analysis', '')
            workout.feedback = analysis.get('feedback', '')
            
            db.session.add(workout)
            db.session.commit()
            
            return redirect(url_for('session_detail', session_id=workout.id))
            
        except Exception as e:
            os.remove(temp_path)
            flash('Error processing video')
            return redirect(request.url)
            
    return redirect(url_for('dashboard'))

def analyze_workout_video(video_path, notes=None):
    """
    Analyze a workout video and return exercise details and feedback.
    """
    try:
        # Check video duration
        clip = VideoFileClip(video_path)
        if clip.duration > 30:
            clip.close()
            os.remove(video_path)
            raise ValueError("Video must be 30 seconds or shorter")
        clip.close()
        
        # Get exercise type from the request
        exercise_type = request.form.get('exercise_type', 'unspecified exercise')
        
        # Use the process_video function to get analysis
        result = process_video(video_path, exercise_type, notes)
        
        return {
            'exercise': result['exercise'],
            'analysis': result['analysis'],
            'feedback': result['feedback'],
            'character_id': result['character_id'],
            'character_name': result['character_name']
        }
    except Exception as e:
        print(f"Error in analyze_workout_video: {str(e)}")  # Add debugging
        # Clean up the video file if there's an error
        if os.path.exists(video_path):
            os.remove(video_path)
        raise e

def analyze_workout_image(image_path, notes=None):
    """
    Analyze a workout image and return exercise details and feedback.
    """
    try:
        # For now, return a simple analysis
        exercise = "general form check"
        analysis = "Image-based analysis is limited. Consider uploading a video for more detailed feedback."
        feedback = generate_feedback(exercise, analysis)
        
        return {
            'exercise': exercise,
            'analysis': analysis,
            'feedback': feedback,
            'character_id': 'trainer',
            'character_name': 'Personal Trainer'
        }
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise e

@app.route('/register', methods=['GET', 'POST'])
def register():
    current_hour = datetime.now().hour
    
    if 0 <= current_hour < 11:
        greeting = "Good morning"
    elif 11 <= current_hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if not username or not email or not password:
            flash("Username, Email, and Password required", "error")
            return redirect(url_for('register'))

        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists", "error")
            return redirect(url_for('register'))

        # Check if email already registered
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash("Email already registered", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful, please login", "success")
        return redirect(url_for('login'))
    return render_template('register.html', greeting=greeting)

@app.route('/login', methods=['GET', 'POST'])
def login():
    current_hour = datetime.now().hour
    
    if 0 <= current_hour < 11:
        greeting = "Good morning"
    elif 11 <= current_hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            next_page = request.form.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('home')
            return redirect(next_page)
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for('login'))
    return render_template('login.html', greeting=greeting)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/exercise-library')
def exercise_library():
    return render_template('exercise_library.html')

@app.route('/form-guidelines', methods=['GET', 'POST'])
def form_guidelines():
    exercises = {
        'squat': {
            'title': 'Squat Form Guidelines',
            'key_points': [
                'Keep your feet shoulder-width apart',
                'Keep your chest up and core tight',
                'Push your hips back as if sitting in a chair',
                'Keep your knees in line with your toes',
                'Go as low as you can while maintaining form'
            ],
            'common_mistakes': [
                'Knees caving inward',
                'Rounding the back',
                'Heels coming off the ground',
                'Not going deep enough'
            ],
            'videos': [
                {'title': 'Perfect Squat Form Guide', 'url': 'https://www.youtube.com/watch?v=ultWZbUMPL8'},
                {'title': 'Common Squat Mistakes', 'url': 'https://www.youtube.com/watch?v=FQKfr1YDhEk'}
            ]
        },
        'deadlift': {
            'title': 'Deadlift Form Guidelines',
            'key_points': [
                'Position the bar over mid-foot',
                'Bend at hips and knees to grip the bar',
                'Keep your chest up and back straight',
                'Keep the bar close to your body',
                'Drive through your heels'
            ],
            'common_mistakes': [
                'Rounding the back',
                'Bar too far from shins',
                'Starting with hips too low',
                'Not engaging lats'
            ],
            'videos': [
                {'title': 'Deadlift Tutorial', 'url': 'https://www.youtube.com/watch?v=wYREQkVtvEc'},
                {'title': 'Fix Your Deadlift', 'url': 'https://www.youtube.com/watch?v=NYN3UGCYisk'}
            ]
        },
        'bench_press': {
            'title': 'Bench Press Form Guidelines',
            'key_points': [
                'Plant feet firmly on the ground',
                'Keep your back arched',
                'Grip slightly wider than shoulder-width',
                'Lower the bar to mid-chest',
                'Keep elbows at 45-degree angle'
            ],
            'common_mistakes': [
                'Bouncing the bar off chest',
                'Elbows flaring too wide',
                'Not maintaining arch',
                'Feet moving during lift'
            ],
            'videos': [
                {'title': 'Perfect Bench Press', 'url': 'https://www.youtube.com/watch?v=vcBig73ojpE'},
                {'title': 'Bench Press Mistakes', 'url': 'https://www.youtube.com/watch?v=vthMCtgVtFw'}
            ]
        }
    }
    
    if request.method == 'POST':
        exercise = request.form.get('exercise')
        if exercise in exercises:
            return render_template('form_guidelines.html', 
                                exercise_data=exercises[exercise],
                                exercises=exercises.keys())
    
    return render_template('form_guidelines.html', 
                         exercises=exercises.keys(),
                         exercise_data=None)

@app.route('/previous-analyses')
@login_required
def previous_analyses():
    sessions = WorkoutSession.query.filter_by(user_id=current_user.id).order_by(WorkoutSession.created_at.desc()).all()
    return render_template('previous_analyses.html', sessions=sessions)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/characters')
@login_required
def characters():
    current_character = session.get('character_id', 'trainer')
    return render_template('characters.html', 
                         characters=CHARACTERS.values(),
                         current_character=current_character)

@app.route('/set-character', methods=['POST'])
@login_required
def set_character():
    data = request.get_json()
    character_id = data.get('character_id')
    if character_id in CHARACTERS:
        session['character_id'] = character_id
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/delete-session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    session = WorkoutSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if session:
        # Delete the video file if it exists
        if session.video_filename:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], session.video_filename)
            if os.path.exists(video_path):
                os.remove(video_path)
        
        # Delete the database record
        db.session.delete(session)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

_table_created = False

@app.before_request
def ensure_tables():
    global _table_created
    if not _table_created:
        db.create_all()
        _table_created = True

@app.context_processor
def utility_processor():
    return {'CHARACTERS': CHARACTERS}

if __name__ == '__main__':
    # This block will run when executing "python app.py"
    # With the before_first_request in place, tables will also be created when running with "flask run"
    with app.app_context():
        db.create_all()
    # Run the Flask development server
    app.run(debug=True)
