from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import mediapipe as mp
from simple_facerec import SimpleFacerec

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

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

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

def detect_emotion(frame, x=None, y=None, w=None, h=None):
    """
    Detect emotion using FER and fallback to simple smile detection if FER fails
    
    Args:
        frame: The full frame image
        x, y, w, h: Face region coordinates (optional, used for fallback)
    """
    # Try using the FER emotion detector if available
    if FER_AVAILABLE and emotion_detector is not None:
        try:
            emotion_result = emotion_detector.top_emotion(frame)
            if emotion_result and emotion_result[0]:
                emotion_name = emotion_result[0]
                # Map emotion names to emojis
                emotion_map = {
                    "happy": "Happy 😊",
                    "sad": "Sad 😢",
                    "angry": "Angry 😠",
                    "fear": "Fearful 😨",
                    "neutral": "Neutral 😐",
                    "surprise": "Surprised 😲",
                    "disgust": "Disgusted 🤢"
                }
                return emotion_map.get(emotion_name, f"{emotion_name.capitalize()} 😐")
        except Exception as e:
            print(f"FER error: {e}")
    
    # Use simple smile detection as fallback
    # Extract face region if coordinates are provided
    if all(coord is not None for coord in [x, y, w, h]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return "Happy 😊"
        else:
            return "Sad 😢"
    else:
        # Try to detect faces first if coordinates aren't provided
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            if len(smiles) > 0:
                return "Happy 😊"
            else:
                return "Sad 😢"
    
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
    global attention_state, emotion_state
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
        attention_state = detect_attention(frame)
        
        # Break suggestion based on emotion
        if emotion_label.startswith(("Sad", "Angry")) or attention_state == "Distracted":
            cv2.putText(frame, "💤 Break Time!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        
        cv2.putText(frame, f"Attention: {attention_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/face_recognition")
def face_recognition_endpoint():
    """
    Returns the face recognition results from the current frame
    """
    ret, frame = camera.read()
    if not ret:
        return JSONResponse({'error': 'No frame'})
    
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        faces = []
        for (y1, x2, y2, x1), name in zip(face_locations, face_names):
            faces.append({
                "name": name,
                "location": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "recognized": name != "Unknown"
            })
        
        return {
            "faces": faces,
            "count": len(faces)
        }
    except Exception as e:
        return JSONResponse({'error': str(e)})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/status")
def status():
    global distracted_or_sad_start
    ret, frame = camera.read()
    if not ret:
        return JSONResponse({'error': 'No frame'})
    
    # Get emotion using FER
    emotion = detect_emotion(frame)
    
    # Face recognition
    face_info = {"recognized": False, "name": "Unknown"}
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        if face_locations and face_names:
            face_info = {
                "recognized": face_names[0] != "Unknown",
                "name": face_names[0]
            }
    except Exception:
        # Fallback to simple face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_info["detected"] = len(faces) > 0
    
    # Attention detection
    attention = detect_attention(frame)
    
    # Break suggestion logic
    current_time = time.time()
    should_break = False
    is_negative = emotion.startswith(("Sad", "Angry")) or attention == "Distracted"
    
    if is_negative:
        if distracted_or_sad_start is None:
            distracted_or_sad_start = current_time
        elif current_time - distracted_or_sad_start >= break_threshold:
            should_break = True
    else:
        distracted_or_sad_start = None
    
    return {
        "emotion": emotion,
        "attention": attention,
        "face": face_info,
        "should_break": should_break
    }

@app.get("/emotion_only")
def emotion_only():
    """
    Returns just the emotion detection results from FER
    """
    ret, frame = camera.read()
    if not ret:
        return JSONResponse({'error': 'No frame'})
    
    if not FER_AVAILABLE or emotion_detector is None:
        # Fallback to simple smile detection if FER is not available
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            emotion = "Happy 😊" if len(smiles) > 0 else "Sad 😢"
            
            return {
                "emotions": {
                    "happy": 1.0 if len(smiles) > 0 else 0.0,
                    "sad": 0.0 if len(smiles) > 0 else 1.0
                },
                "top_emotion": "happy" if len(smiles) > 0 else "sad",
                "confidence": 0.9,
                "note": "Using fallback smile detection as FER is not available"
            }
        else:
            return {"message": "No face detected for emotion analysis"}
    
    # FER is available, use it
    try:
        # Get raw emotion data from FER
        emotion_data = emotion_detector.detect_emotions(frame)
        top_emotion = emotion_detector.top_emotion(frame)
        
        if emotion_data and len(emotion_data) > 0:
            return {
                "emotions": emotion_data[0]["emotions"],
                "top_emotion": top_emotion[0] if top_emotion else None,
                "confidence": top_emotion[1] if top_emotion else None
            }
        else:
            return {"message": "No face detected for emotion analysis"}
    except Exception as e:
        return JSONResponse({'error': str(e)})
