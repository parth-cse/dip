import os
import cv2

# Set parameters
DATA_DIR = './marathi_data'
MARATHI_LETTERS = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ']
dataset_size = 100  

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

index = 0
for letter in MARATHI_LETTERS:
    letter_dir = os.path.join(DATA_DIR, index.__str__())
    os.makedirs(letter_dir, exist_ok=True)

    print(f'\nCollecting data for: {letter}')

  
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read properly.")
            continue

        cv2.putText(frame, f'Ready to collect "{letter}" - Press "q" to start',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        file_path = os.path.join(letter_dir, f'{counter}.jpg')
        saved = cv2.imwrite(file_path, frame)

        if saved:
            print(f'Saved: {file_path}')
        else:
            print(f'Failed to save: {file_path}')

        cv2.putText(frame, f'Collecting "{letter}": {counter}/{dataset_size}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Data Collection', frame)

        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('e'): 
            break
    index += 1
    print(f'Finished collecting data for: {letter}')

cap.release()
cv2.destroyAllWindows()

print("\nData collection complete.")