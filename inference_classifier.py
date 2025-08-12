import cv2
import pickle
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import time
from threading import Thread

# Load trained model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Label map
label_map = {
    '0': 'आ', '1': 'र', '2': 'ती', '3': 'वि', '4': 'शा',
    '5': 'ल'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Tkinter setup
root = tk.Tk()
root.title("Marathi Gesture Recognition")
root.geometry("850x650")

frame_label = tk.Label(root)
frame_label.pack()

prediction_label = tk.Label(root, text="Prediction: --", font=("Arial", 20), fg="green")
prediction_label.pack(pady=10)

output_label = tk.Label(root, text="", font=("Arial", 24), fg="blue")
output_label.pack(pady=10)

start_button = tk.Button(root, text="Start Recognition", font=("Arial", 16), bg="green", fg="white")
start_button.pack(pady=10)

# Globals
recognized_word = ""
last_prediction = None
last_time = time.time()
cap = None
recognizing = False

def start_recognition():
    global cap, recognizing, recognized_word, last_prediction, last_time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        prediction_label.config(text="Webcam not found", fg="red")
        return

    recognizing = True
    recognized_word = ""
    last_prediction = None
    last_time = time.time()
    start_button.config(state='disabled')
    update_video()

def update_video():
    global recognized_word, last_prediction, last_time

    if not recognizing:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_video)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_, y_, data_aux = [], [], []

        # ✅ FIXED protobuf list addition
        all_landmarks = list(results.multi_hand_landmarks[0].landmark) + list(results.multi_hand_landmarks[1].landmark)

        for lm in all_landmarks:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in all_landmarks:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        current_time = time.time()
        if current_time - last_time >= 2:
            if len(data_aux) == 84:
                prediction = model.predict([data_aux])[0]
                letter = label_map.get(str(prediction), "?")

                if letter != last_prediction:
                    last_prediction = letter
                    recognized_word += letter
                    output_label.config(text=recognized_word)
                    prediction_label.config(text=f"Prediction: {letter}")
                    last_time = current_time
            else:
                prediction_label.config(text="⚠️ Incomplete hand landmarks", fg="orange")

    # Update frame in GUI
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)

    root.after(10, update_video)

def handle_key(event):
    global recognized_word, recognizing
    if event.keysym == 'space':
        recognized_word += " "
        output_label.config(text=recognized_word)
    elif event.keysym == 'Escape':
        recognizing = False
        if cap:
            cap.release()
        output_label.config(text=recognized_word)
        prediction_label.config(text="Recognition Stopped")
        start_button.config(state='normal')

root.bind('<space>', handle_key)
root.bind('<Escape>', handle_key)
start_button.config(command=start_recognition)

root.mainloop()