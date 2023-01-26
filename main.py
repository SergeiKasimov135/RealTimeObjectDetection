import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

window = tk.Tk()
window.geometry("480x700")
window.title("BodyApp")
ck.set_appearance_mode("dark")


frame = tk.Frame(height="480", width="480")
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)


with open("pickle.pkl", "rb") as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(0)
current_stage = ""
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ""


def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), 
                                                      thickness=2,
                                                      circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), 
                                                      thickness=2,
                                                      circle_radius=2))

    try:
        pass
    except Exception as e:
        pass

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imagetk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    
detect()
window.mainloop()
