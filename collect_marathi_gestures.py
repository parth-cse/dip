import os
import time
import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread

# Parameters
DATA_DIR = './marathi_data'
MARATHI_LETTERS = ['आ', 'र', 'ती', 'वि', 'शा', 'ल']
dataset_size = 100

# Create dataset directory
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Tkinter GUI
root = tk.Tk()
root.title("Marathi Gesture Data Collection")
root.geometry("800x600")
root.resizable(False, False)

status_label = tk.Label(root, text="Initializing...", font=("Arial", 24), fg="blue")
status_label.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

# Global control flags
start_letter = False
skip_letter = False
current_letter = ""
frame = None  # Global frame for display

# Video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def update_ui(text, color="black"):
    status_label.config(text=text, fg=color)
    root.update_idletasks()

# Key bindings: Enter = start, Escape = skip
def on_key_press(event):
    global start_letter, skip_letter
    if event.keysym == 'Return':
        start_letter = True
    elif event.keysym == 'Escape':
        skip_letter = True

root.bind("<Key>", on_key_press)

# Main capture loop in background thread
def capture_and_process():
    global start_letter, skip_letter, frame
    index = 0

    for letter in MARATHI_LETTERS:
        start_letter = False
        skip_letter = False
        current_letter = letter
        counter = 0
        letter_dir = os.path.join(DATA_DIR, str(index))
        os.makedirs(letter_dir, exist_ok=True)

        update_ui(f'Ready to collect: "{letter}" — Press ENTER to start', "green")
        while not start_letter:
            time.sleep(0.01)

        update_ui(f'Collecting "{letter}"...', "orange")

        while counter < dataset_size and not skip_letter:
            ret, live_frame = cap.read()
            if not ret:
                continue

            flipped = cv2.flip(live_frame, 1)
            frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                # Save clean frame (without drawing)
                clean_frame = flipped.copy()
                file_path = os.path.join(letter_dir, f'{counter}.jpg')
                cv2.imwrite(file_path, clean_frame)

                # Draw landmarks only for display
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                print(f"Saved {file_path}")
                counter += 1
                update_ui(f'Collecting "{letter}": {counter}/{dataset_size}', "orange")

            # Show updated frame with landmarks in GUI
            frame_rgb_display = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            img_display = Image.fromarray(frame_rgb_display)
            imgtk = ImageTk.PhotoImage(image=img_display)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

            time.sleep(0.01)  # Reduce CPU usage and prevent flicker

        if skip_letter:
            update_ui(f'Skipped: "{letter}"', "gray")
        else:
            update_ui(f'Finished: "{letter}" ✅', "green")
        index += 1

    # After all letters collected or skipped
    cap.release()
    hands.close()

    # Hide video frame
    video_label.config(image='')

    update_ui("✅ All data collected!", "blue")

    # Thank you message and exit button
    thank_you_label = tk.Label(root, text="Thank you for contributing!", font=("Arial", 28), fg="darkgreen")
    thank_you_label.pack(pady=30)

    exit_button = tk.Button(root, text="Exit", font=("Arial", 18), bg="red", fg="white", command=root.destroy)
    exit_button.pack(pady=10)

# Start background thread only (no show_frame now)
Thread(target=capture_and_process, daemon=True).start()
root.mainloop()
