import os
import pickle
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from threading import Thread

# === Configuration ===
DATA_DIR = './marathi_data'
INDEX_TO_LETTER = ['आ', 'र', 'ती', 'वि', 'शा', 'ल']

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# === Tkinter GUI Setup ===
root = tk.Tk()
root.title("Marathi Gesture Data Preprocessing")
root.geometry("600x400")
root.resizable(False, False)

status_label = tk.Label(root, text="Click 'Process Data' to begin", font=("Arial", 16))
status_label.pack(pady=10)

letter_label = tk.Label(root, text="Letter: --", font=("Arial", 20), fg="purple")
letter_label.pack(pady=5)

count_label = tk.Label(root, text="Valid: 0 | Skipped: 0", font=("Arial", 14))
count_label.pack()

progress = ttk.Progressbar(root, length=400, mode='determinate')
progress.pack(pady=20)

process_button = tk.Button(root, text="📦 Process Data", font=("Arial", 16), bg="green", fg="white")
process_button.pack()

# === Global State ===
data = []
labels = []
valid_count = 0
skipped_count = 0

# === Main Processing Function ===
def process_data():
    global valid_count, skipped_count, data, labels
    process_button.config(state="disabled")
    status_label.config(text="Processing...", fg="blue")
    root.update_idletasks()

    label_folders = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
    total_letters = len(label_folders)

    for folder in label_folders:
        try:
            label_index = int(folder)
            letter = INDEX_TO_LETTER[label_index]
        except (ValueError, IndexError):
            continue

        folder_path = os.path.join(DATA_DIR, folder)
        images = os.listdir(folder_path)

        letter_label.config(text=f"Letter: {letter}")
        progress["maximum"] = len(images)
        progress["value"] = 0
        root.update_idletasks()

        for i, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                skipped_count += 1
                count_label.config(text=f"Valid: {valid_count} | Skipped: {skipped_count}")
                progress["value"] = i + 1
                root.update_idletasks()
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Require exactly 2 hands
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
                skipped_count += 1
                count_label.config(text=f"Valid: {valid_count} | Skipped: {skipped_count}")
                progress["value"] = i + 1
                root.update_idletasks()
                continue

            x_, y_ = [], []
            data_aux = []

            all_landmarks = list(results.multi_hand_landmarks[0].landmark) + list(results.multi_hand_landmarks[1].landmark)

            for lm in all_landmarks:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in all_landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # 42*2 = 84 features expected
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(label_index)
                valid_count += 1
                count_label.config(text=f"Valid: {valid_count} | Skipped: {skipped_count}")
            else:
                skipped_count += 1

            progress["value"] = i + 1
            root.update_idletasks()

    # Save extracted data
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    with open('label_map.pickle', 'wb') as f:
        pickle.dump(INDEX_TO_LETTER, f)

    status_label.config(text="✅ Data processing complete. Saved to 'data.pickle'", fg="green")
    process_button.config(state="normal")

# === Threaded Start ===
def run_thread():
    Thread(target=process_data, daemon=True).start()

process_button.config(command=run_thread)
root.mainloop()