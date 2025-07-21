from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load cascades for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Mediapipe FaceMesh with iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Attention & Emotion state tracking
last_attention_time = time.time()
last_emotion_time = time.time()
attention_timeout = 5  # seconds
attention_state = "Attentive"
emotion_state = "Neutral 😐"

# Duration monitoring
distracted_or_sad_start = None
break_threshold = 5 * 60  # 5 minutes


def detect_emotion(gray, x, y, w, h):
    roi_gray = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    if len(smiles) > 0:
        return "Happy 😊"
    else:
        return "Sad 😢"


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
            return np.linalg.norm(np.array(iris) - np.array(inner)) / \
                   (np.linalg.norm(np.array(outer) - np.array(inner)) + 1e-6)

        left_ratio = gaze_ratio(left_iris, left_eye_outer, left_eye_inner)
        right_ratio = gaze_ratio(right_iris, right_eye_outer, right_eye_inner)

        vertical_iris_shift = abs(left_iris[1] - left_eye_top[1]) + abs(right_iris[1] - right_eye_top[1])

        # Distraction logic
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        emotion_label = "Neutral 😐"

        for (x, y, w, h) in faces:
            emotion_label = detect_emotion(gray, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        emotion_state = emotion_label
        attention_state = detect_attention(frame)

        cv2.putText(frame, f"Attention: {attention_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('detect.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    global distracted_or_sad_start
    ret, frame = camera.read()
    if not ret:
        return jsonify({'error': 'No frame'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion = "Neutral 😐"
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        emotion = detect_emotion(gray, x, y, w, h)
    else:
        emotion = "Neutral 😐"

    attention = detect_attention(frame)

    # Update global state
    current_time = time.time()
    should_break = False

    if emotion == "Sad 😢" or attention == "Distracted":
        if distracted_or_sad_start is None:
            distracted_or_sad_start = current_time
        elif current_time - distracted_or_sad_start >= break_threshold:
            should_break = True
    else:
        distracted_or_sad_start = None

    return jsonify({
        "emotion": emotion,
        "attention": attention,
        "should_break": should_break
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

