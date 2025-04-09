import cv2
import pickle
import mediapipe as mp
import numpy as np

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

label_map = {
    '0': 'अ', '1': 'आ', '2': 'इ', '3': 'ई', '4': 'उ',
    '5': 'ऊ', '6': 'ए', '7': 'ऐ', '8': 'ओ', '9': 'औ'
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("🎥 Starting real-time prediction. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction_text = 'No hand detected'

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict([data_aux])[0]
            prediction_text = f'Predicted: {label_map.get(str(prediction), "?")}'
            print(prediction_text)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, prediction_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Marathi Sign Classifier", frame)

    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
