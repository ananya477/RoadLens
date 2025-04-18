# Road Sign Detection Web Application

A web application that allows users to upload road sign images and detect them using a trained model.

## Features

- User authentication (signup/login)
- Vehicle information storage
- Road sign image/dashcam video upload
- Road sign detection using AI model

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the uploads directory:
```bash
mkdir static/uploads
```

4. Place your trained model file (best.pt) in the root directory

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Sign up for a new account
2. Enter your vehicle information
3. Log in to your account
4. Upload a road sign image/dashcam video
5. View the detection results

## Note

Make sure to place your trained model file (best.pt) in the root directory before running the application. The model will be used for road sign detection when users upload images. 
