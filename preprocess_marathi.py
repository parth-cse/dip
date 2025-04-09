import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './marathi_data'

data = []
labels = []

valid_count = 0
skipped_count = 0

for label in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {img_path}")
            skipped_count += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"\nNo hand detected in {img_path}")
            skipped_count += 1
            continue

        # Extract landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_)) 
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(int(label)) 
            valid_count += 1

print(f"\nFinished preprocessing.")
print(f"Valid samples: {valid_count}")
print(f"Skipped samples: {skipped_count}")

# Save to file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Saved processed data to 'data.pickle'")