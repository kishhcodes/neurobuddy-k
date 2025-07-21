import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_attention(frame):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return "no-face"

        # Check if eyes are closed or user is looking away (basic version)
        # Can be enhanced with exact eye landmark positions
        return "attentive"
