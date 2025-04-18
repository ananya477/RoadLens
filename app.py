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
import numpy as np
import torch

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
                                 result=result['message'],
                                 image_path=filename,
                                 bbox=result['bbox'],
                                 class_name=result['class_name'],
                                 confidence=result['confidence'])
    
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame skip based on video length and FPS
            frame_skip = max(1, min(total_frames // 800, fps // 10))  # Less aggressive frame skipping
            
            # Create output video writer with H.264 codec
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['VIDEO_FOLDER'], output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                # Fallback to XVID if H.264 is not available
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_filename = f"processed_{os.path.splitext(filename)[0]}.avi"
                output_path = os.path.join(app.config['VIDEO_FOLDER'], output_filename)
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                if not out.isOpened():
                    flash('Error: Could not create output video')
                    return redirect(request.url)
            
            detections = []
            batch_frames = []
            batch_size = 8  # Reduced batch size
            frame_count = 0  # Initialize frame counter
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Resize frame for processing
                frame = cv2.resize(frame, (640, 640))
                batch_frames.append(frame)
                
                if len(batch_frames) == batch_size:
                    # Process batch
                    results = model_handler.model.predict(
                        source=batch_frames,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        device=model_handler.device,
                        half=True,
                        agnostic_nms=True,
                        max_det=50,
                        stream=True
                    )
                    
                    # Process results and write frames
                    for i, result in enumerate(results):
                        frame = batch_frames[i]
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = f"{model_handler.model.names[cls]} {conf:.2f}"
                            
                            # Draw bounding box and label
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Write processed frame
                        out.write(frame)
                    
                    batch_frames = []
                
                frame_count += 1
            
            # Process remaining frames
            if batch_frames:
                results = model_handler.model.predict(
                    source=batch_frames,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    device=model_handler.device,
                    half=True,
                    agnostic_nms=True,
                    max_det=50,
                    stream=True
                )
                
                # Process results and write frames
                for i, result in enumerate(results):
                    frame = batch_frames[i]
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{model_handler.model.names[cls]} {conf:.2f}"
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Write processed frame
                    out.write(frame)
            
            cap.release()
            out.release()
            processing_time = time.time() - start_time
            
            return render_template('video_result.html',
                                 original_video=filename,
                                 processed_video=output_filename,
                                 processing_time=f"{processing_time:.1f}")
    
    return render_template('video_upload.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000) 