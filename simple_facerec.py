import face_recognition
import os
import cv2
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        images_path_list = os.listdir(images_path)

        for img_name in images_path_list:
            img_path = os.path.join(images_path, img_name)
            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(os.path.splitext(img_name)[0])

        print("✅ Encoding Complete")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_locations_final = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            face_locations_final.append((y1, x2, y2, x1))

        return face_locations_final, face_names
