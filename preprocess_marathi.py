import os
import pickle
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from threading import Thread
import random

# === Configuration ===
DATA_DIR = './data'
CONSONANT_DIR = os.path.join(DATA_DIR, 'consonant')
VOWEL_DIR = os.path.join(DATA_DIR, 'vowel')
MATRA_DIR = os.path.join(DATA_DIR, 'matra')

vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ']
matras = ['', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ']
consonants = [
    'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ',
    'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब',
    'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह', 'ळ',
    'क्ष', 'ज्ञ'
]

# Final index mapping: vowels + consonants + matras + (consonant + matra)
INDEX_TO_LETTER = vowels + consonants + matras[1:] + [c + m for c in consonants for m in matras if m]

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

    total_letters = len(INDEX_TO_LETTER)

    for label_index, letter in enumerate(INDEX_TO_LETTER):
        letter_label.config(text=f"Letter: {letter}")
        root.update_idletasks()

        # 1-hand gestures: vowels, consonants, matras
        if letter in vowels:
            folder_path = os.path.join(VOWEL_DIR, str(vowels.index(letter)))
        elif letter in consonants:
            folder_path = os.path.join(CONSONANT_DIR, str(vowels.__len__() + consonants.index(letter)))
        elif letter in matras[1:]:
            folder_path = os.path.join(MATRA_DIR, str(matras.index(letter)-1))
        else:
            folder_path = None

        if folder_path:
            if not os.path.exists(folder_path):
                continue

            images = os.listdir(folder_path)
            progress["maximum"] = len(images)
            progress["value"] = 0

            for i, img_name in enumerate(images):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 1:
                    skipped_count += 1
                    continue

                x_, y_, data_aux = [], [], []
                for lm in results.multi_hand_landmarks[0].landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in results.multi_hand_landmarks[0].landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(label_index)
                    valid_count += 1
                else:
                    skipped_count += 1

                count_label.config(text=f"Valid: {valid_count} | Skipped: {skipped_count}")
                progress["value"] = i + 1
                root.update_idletasks()

        # 2-hand combo: consonant + matra
        elif len(letter) > 1:
            consonant, matra = letter[0], letter[1:]

            con_index = vowels.__len__() + consonants.index(consonant)
            matra_vowel = vowels[matras.index(matra)]
            matra_index = vowels.index(matra_vowel)

            con_path = os.path.join(CONSONANT_DIR, str(con_index))
            matra_path = os.path.join(VOWEL_DIR, str(matra_index))

            if not os.path.exists(con_path) or not os.path.exists(matra_path):
                skipped_count += 1
                continue

            con_imgs = os.listdir(con_path)
            matra_imgs = os.listdir(matra_path)

            min_len = min(len(con_imgs), len(matra_imgs))
            progress["maximum"] = min_len

            for i in range(min_len):
                con_img = cv2.imread(os.path.join(con_path, con_imgs[i]))
                matra_img = cv2.imread(os.path.join(matra_path, matra_imgs[i]))

                if con_img is None or matra_img is None:
                    skipped_count += 1
                    continue

                combined_img = cv2.hconcat([con_img, matra_img])
                combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
                results = hands.process(combined_rgb)

                if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
                    skipped_count += 1
                    continue

                x_, y_, data_aux = [], [], []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                if len(data_aux) == 84:
                    data.append(data_aux)
                    labels.append(label_index)
                    valid_count += 1
                else:
                    skipped_count += 1

                count_label.config(text=f"Valid: {valid_count} | Skipped: {skipped_count}")
                progress["value"] = i + 1
                root.update_idletasks()

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
