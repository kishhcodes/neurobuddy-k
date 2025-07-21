import cv2
from simple_facerec import SimpleFacerec
from fer import FER

# Load face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images("known_faces/")

# Load emotion detector
emotion_detector = FER(mtcnn=True)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Recognize face
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Detect emotion (only on full frame for simplicity)
    emotion_result = emotion_detector.top_emotion(frame)
    emotion = emotion_result[0] if emotion_result else "No Emotion"

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc

        # Draw face box & name
        cv2.putText(frame, f"{name}", (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show emotion near face
        cv2.putText(frame, f"{emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

        # Break trigger
        if emotion in ["sad", "angry"]:
            cv2.putText(frame, "💤 Break Time!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Face + Emotion", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
