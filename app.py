from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from PIL import Image
from werkzeug.utils import secure_filename
from model_handler import ModelHandler
import cv2
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VIDEO_FOLDER'] = 'static/videos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the model handler
model_handler = ModelHandler()

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    vehicle = db.relationship('Vehicle', backref='owner', uselist=False)

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('vehicle_info'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('select_input'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/vehicle_info', methods=['GET', 'POST'])
@login_required
def vehicle_info():
    if request.method == 'POST':
        model = request.form.get('model')
        name = request.form.get('name')
        
        vehicle = Vehicle(model=model, name=name, user_id=current_user.id)
        db.session.add(vehicle)
        db.session.commit()
        logout_user()
        return redirect(url_for('login'))
    
    return render_template('vehicle_info.html')

@app.route('/select_input')
@login_required
def select_input():
    return render_template('select_input.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run prediction using the model handler
            result = model_handler.predict(filepath)
            
            # Pass the filename instead of the full path
            return render_template('result.html', 
                                 result=result, 
                                 image_path=filename)
    
    return render_template('upload.html')

@app.route('/video_upload', methods=['GET', 'POST'])
@login_required
def video_upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file')
            return redirect(request.url)
        
        video = request.files['video']
        if video.filename == '':
            flash('No selected video')
            return redirect(request.url)
        
        if video and allowed_video_file(video.filename):
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
            video.save(video_path)
            
            # Process video
            start_time = time.time()
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create output video writer with H.264 codec
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['VIDEO_FOLDER'], output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                # Fallback to XVID if H.264 is not available
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = os.path.join(app.config['VIDEO_FOLDER'], f"processed_{os.path.splitext(filename)[0]}.avi")
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                if not out.isOpened():
                    flash('Error: Could not create output video')
                    return redirect(request.url)
            
            detections = []
            total_signs = 0
            total_confidence = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection directly on the frame
                results = model_handler.model.predict(source=frame, conf=0.25)
                
                # Process detections
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = model_handler.model.names[class_id]
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} ({confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Update statistics
                        total_signs += 1
                        total_confidence += confidence
                        
                        # Add to detections list
                        detections.append({
                            'time': f"{cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.1f}s",
                            'sign_type': class_name,
                            'confidence': f"{confidence:.1f}"
                        })
                
                # Write frame to output video
                out.write(frame)
            
            cap.release()
            out.release()
            processing_time = time.time() - start_time
            avg_confidence = (total_confidence / total_signs) if total_signs > 0 else 0
            
            # Pass both original and processed video paths
            return render_template('video_result.html',
                                 video_path=output_path,
                                 original_video=filename,
                                 total_signs=total_signs,
                                 processing_time=f"{processing_time:.1f}",
                                 avg_confidence=f"{avg_confidence:.1f}",
                                 detections=detections)
    
    return render_template('video_upload.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 