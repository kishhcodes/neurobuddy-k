from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import mediapipe as mp
from simple_facerec import SimpleFacerec
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Try to import FER, but provide a fallback if it fails
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("Warning: FER module could not be imported. Using fallback emotion detection.")
    FER_AVAILABLE = False

app = FastAPI(title="Face Attention & Emotion API", description="Detects face attention and emotion with Swagger UI.")

# CORS (optional, for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
templates = Jinja2Templates(directory="templates")


# Initialize face recognition from SimpleFacerec
sfr = SimpleFacerec()
try:
    sfr.load_encoding_images("known_faces/")
except Exception as e:
    print(f"Warning: Could not load face encodings: {e}")

# Initialize FER emotion detector if available
if FER_AVAILABLE:
    try:
        emotion_detector = FER(mtcnn=True)
    except Exception as e:
        print(f"Error initializing FER: {e}")
        FER_AVAILABLE = False
else:
    emotion_detector = None

last_attention_time = time.time()
attention_timeout = 5
attention_state = "Attentive"
emotion_state = "Neutral 😐"
distracted_or_sad_start = None
break_threshold = 5 * 60
# New variable to track continuous distraction time
distraction_start_time = None
distraction_threshold = 10  # 1 minute in seconds
needs_redirect = False

@app.get("/watch", response_class=HTMLResponse)
def watch_feed(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/break_page", response_class=HTMLResponse)
def break_page(request: Request):
    """Page to redirect to when user is distracted for too long"""
    return templates.TemplateResponse("break.html", {"request": request})

# ... existing code ...

def detect_emotion(frame):
    """Detects emotion from a frame using FER if available, otherwise returns a fallback."""
    if FER_AVAILABLE and emotion_detector is not None:
        try:
            result = emotion_detector.detect_emotions(frame)
            if result and len(result) > 0:
                emotions = result[0]["emotions"]
                if emotions:
                    # Get the emotion with the highest score
                    emotion, score = max(emotions.items(), key=lambda x: x[1])
                    # Add emoji for some common emotions
                    emoji_map = {
                        "happy": "😊",
                        "sad": "😢",
                        "angry": "😠",
                        "surprise": "😲",
                        "neutral": "😐",
                        "fear": "😨",
                        "disgust": "🤢"
                    }
                    emoji = emoji_map.get(emotion.lower(), "")
                    return f"{emotion.capitalize()} {emoji}"
        except Exception as e:
            print(f"Emotion detection error: {e}")
    # Fallback if FER is not available or detection fails
    return "Neutral 😐"

def detect_attention(frame):
    global last_attention_time
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        def get_point(idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        left_eye_outer = get_point(33)
        left_eye_inner = get_point(133)
        right_eye_outer = get_point(362)
        right_eye_inner = get_point(263)
        left_iris = get_point(468)
        right_iris = get_point(473)
        left_eye_top = get_point(159)
        left_eye_bottom = get_point(145)
        right_eye_top = get_point(386)
        right_eye_bottom = get_point(374)
        def eye_aspect_ratio(top, bottom, left, right):
            vertical = np.linalg.norm(np.array(top) - np.array(bottom))
            horizontal = np.linalg.norm(np.array(left) - np.array(right))
            return vertical / horizontal if horizontal != 0 else 0
        left_ear = eye_aspect_ratio(left_eye_top, left_eye_bottom, left_eye_outer, left_eye_inner)
        right_ear = eye_aspect_ratio(right_eye_top, right_eye_bottom, right_eye_outer, right_eye_inner)
        def gaze_ratio(iris, outer, inner):
            return np.linalg.norm(np.array(iris) - np.array(inner)) / (np.linalg.norm(np.array(outer) - np.array(inner)) + 1e-6)
        left_ratio = gaze_ratio(left_iris, left_eye_outer, left_eye_inner)
        right_ratio = gaze_ratio(right_iris, right_eye_outer, right_eye_inner)
        vertical_iris_shift = abs(left_iris[1] - left_eye_top[1]) + abs(right_iris[1] - right_eye_top[1])
        if (
            left_ear < 0.15 and right_ear < 0.15 or
            abs(left_ratio - 0.5) > 0.25 or abs(right_ratio - 0.5) > 0.25 or
            vertical_iris_shift > 20
        ):
            if time.time() - last_attention_time > attention_timeout:
                return "Distracted"
        else:
            last_attention_time = time.time()
            return "Attentive"
    return "Distracted"

def gen_frames():
    global attention_state, emotion_state, distraction_start_time, needs_redirect
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        
        # Process frame with FER for emotion detection
        emotion_label = detect_emotion(frame)
        
        # Face recognition with SimpleFacerec
        try:
            face_locations, face_names = sfr.detect_known_faces(frame)
            
            for (y1, x2, y2, x1), name in zip(face_locations, face_names):
                # Draw face box & name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2)
                cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        except Exception as e:
            # Fallback to simple face detection if face recognition fails
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        emotion_state = emotion_label
        current_attention = detect_attention(frame)
        attention_state = current_attention
        
        # Track distraction time and set redirect flag if needed
        current_time = time.time()
        if current_attention == "Distracted":
            if distraction_start_time is None:
                distraction_start_time = current_time
            elif current_time - distraction_start_time >= distraction_threshold:
                needs_redirect = True
                # Add text indicating redirect is imminent
                cv2.putText(frame, "Redirecting to break page...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            distraction_start_time = None
            needs_redirect = False
        
        # Break suggestion based on emotion
        if emotion_label.startswith(("Sad", "Angry")) or attention_state == "Distracted":
            cv2.putText(frame, "💤 Break Time!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            
            # Add distraction timer if we're tracking distraction
            if distraction_start_time is not None:
                seconds_distracted = int(current_time - distraction_start_time)
                cv2.putText(frame, f"Distracted for: {seconds_distracted}s", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Attention: {attention_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ... existing code ...

@app.get("/check_redirect")
def check_redirect():
    """Endpoint for the frontend to check if a redirect is needed"""
    global needs_redirect
    should_redirect = needs_redirect
    if needs_redirect:
        needs_redirect = False  # Reset after frontend checks
    return {"redirect": should_redirect}

@app.get("/video_feed")
def video_feed():
    """Video streaming endpoint"""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
def status():
    """Return the current status of attention and emotion"""
    global attention_state, emotion_state
    # Get the current face data (simplified for this example)
    face_data = {"name": "Unknown", "recognized": False}
    
    # Check if any faces are recognized (this is a simplified version)
    try:
        ret, frame = camera.read()
        if ret:
            face_locations, face_names = sfr.detect_known_faces(frame)
            if face_locations and face_names:
                # Use the first face detected
                face_data = {"name": face_names[0], "recognized": face_names[0] != "Unknown"}
    except Exception as e:
        print(f"Error in status detection: {e}")
    
    # Determine if a break should be suggested
    should_break = False
    if attention_state == "Distracted" or emotion_state.startswith(("Sad", "Angry")):
        should_break = True
    
    return {
        "attention": attention_state,
        "emotion": emotion_state,
        "face": face_data,
        "should_break": should_break
    }

# ... remaining code ...