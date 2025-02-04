from flask import Flask, request, render_template_string, flash, redirect, url_for, render_template, send_from_directory
import openai
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, current_user, logout_user, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv

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
    video_filename = db.Column(db.String(300), nullable=False)
    exercise = db.Column(db.String(50))
    analysis = db.Column(db.Text)
    feedback = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='sessions')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """
    Check if the uploaded file has one of the allowed extensions.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    """
    Simulate video processing and pose estimation.
    
    In a production scenario, this is where you would
    - Extract video frames.
    - Run a pose estimation algorithm (e.g., using MediaPipe, OpenPose, MoveNet).
    - Compare the extracted keypoints to ideal exercise templates.
    
    For this demonstration, we assume the video is of the "squat" exercise and return dummy analysis.
    
    Returns:
        tuple: (exercise, analysis)
    """
    # Dummy data for simulation
    exercise = "squat"
    analysis = (
        "During the descent, the user's left knee deviated inward by 15 degrees. "
        "The user's back started to round during the lower third of the movement."
    )
    return exercise, analysis

def generate_feedback(exercise, analysis):
    """
    Send the pose analysis data to OpenAI's GPT-4 to obtain a friendly feedback message.
    
    Args:
        exercise (str): The name of the exercise.
        analysis (str): The description of form issues derived from video analysis.
    
    Returns:
        str: Generated feedback message.
    """
    prompt = (
        f"A user performed a {exercise}. The analysis of their form indicates the following issues:\n"
        f"{analysis}\n\n"
        "Provide a detailed, friendly explanation of these issues and offer suggestions to improve the form."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable fitness coach."},
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
    return render_template('dashboard.html', sessions=sessions)

@app.route('/session/<int:session_id>')
@login_required
def session_detail(session_id):
    session_record = WorkoutSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session_record:
        return "Session not found or access denied", 404
    return render_template_string('''
      <!doctype html>
      <html>
        <head>
          <meta charset="utf-8">
          <title>Session Details - Workout AI App</title>
          <style>
            :root {
              --background-color: #E3DAC9;
              --text-color: #333;
              --link-color: #0066cc;
              --header-bg: #fff;
              --header-text: #333;
            }
            body.dark-mode {
              --background-color: #222;
              --text-color: #ddd;
              --link-color: #66aaff;
              --header-bg: #333;
              --header-text: #fff;
            }
            body { background-color: var(--background-color); color: var(--text-color); font-family: Arial, sans-serif; margin: 0; padding: 0; }
            header { display: flex; justify-content: space-between; align-items: center; background-color: var(--header-bg); padding: 10px 20px; }
            header h1 { margin: 0; font-size: 20px; color: var(--header-text); }
            .dark-mode-toggle { background: none; border: none; cursor: pointer; font-size: 16px; color: var(--header-text); }
            .container { padding: 20px; }
            a { color: var(--link-color); text-decoration: none; }
            a:hover { text-decoration: underline; }
            ul { list-style-type: none; padding: 0; }
            li { margin-bottom: 10px; }
          </style>
          <script>
            function toggleDarkMode() {
              document.body.classList.toggle('dark-mode');
              if(document.body.classList.contains('dark-mode')){
                  localStorage.setItem('darkMode', 'true');
              } else {
                  localStorage.setItem('darkMode', 'false');
              }
            }
            window.addEventListener('load', function(){
              if(localStorage.getItem('darkMode') === 'true'){
                  document.body.classList.add('dark-mode');
              }
            });
          </script>
        </head>
        <body>
          <header>
            <h1>Workout AI App</h1>
            <button class="dark-mode-toggle" onclick="toggleDarkMode()">Toggle Dark Mode</button>
          </header>
          <div class="container">
            <h2>Session Details</h2>
            <p><strong>Exercise:</strong> {{ session_record.exercise }}</p>
            <p><strong>Analysis:</strong> {{ session_record.analysis }}</p>
            <p><strong>Feedback:</strong><br>{{ session_record.feedback }}</p>
            <p><strong>Uploaded Video:</strong> {{ session_record.video_filename }}</p>
            <p><a href="{{ url_for('dashboard') }}">Back to Dashboard</a></p>
          </div>
        </body>
      </html>
    ''', session_record=session_record)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'video' not in request.files:
        flash("No video file provided", "error")
        return redirect(url_for('home'))
        
    file = request.files['video']
    notes = request.form.get('notes', '')
    
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('home'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Process the video and generate analysis
        exercise, analysis = process_video(video_path)
        feedback = generate_feedback(exercise, analysis)
        
        # Save session with notes
        new_session = WorkoutSession(
            user_id=current_user.id,
            video_filename=filename,
            exercise=exercise,
            analysis=analysis,
            feedback=feedback,
            notes=notes
        )
        db.session.add(new_session)
        db.session.commit()
        
        flash("Video uploaded and analyzed successfully!", "success")
        return redirect(url_for('session_detail', session_id=new_session.id))
    
    flash("Invalid file type", "error")
    return redirect(url_for('home'))

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

_table_created = False

@app.before_request
def ensure_tables():
    global _table_created
    if not _table_created:
        db.create_all()
        _table_created = True

if __name__ == '__main__':
    # This block will run when executing "python app.py"
    # With the before_first_request in place, tables will also be created when running with "flask run"
    with app.app_context():
        db.create_all()
    # Run the Flask development server
    app.run(debug=True)
