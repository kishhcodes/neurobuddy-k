# check_cv2.py
import cv2
print("cv2 version:", cv2.__version__)
print("Has VideoCapture:", hasattr(cv2, "VideoCapture"))
