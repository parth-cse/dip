import pickle
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the INDEX_TO_LETTER mapping
with open('label_map.pickle', 'rb') as f:
    INDEX_TO_LETTER = pickle.load(f)

# Tkinter Setup
root = tk.Tk()
root.title("Marathi Gesture Model Trainer")
root.geometry("700x500")
root.resizable(False, False)

# UI Elements
status_label = tk.Label(root, text="Click 'Train Model' to start", font=("Arial", 16))
status_label.pack(pady=10)

sample_label = tk.Label(root, text="Samples: -- | Classes: --", font=("Arial", 14))
sample_label.pack(pady=5)

progress_bar = ttk.Progressbar(root, length=500, mode='determinate')
progress_bar.pack(pady=10)

train_button = tk.Button(root, text="🧠 Train Model", font=("Arial", 14), bg="blue", fg="white")
train_button.pack(pady=5)

report_box = scrolledtext.ScrolledText(root, width=80, height=15, font=("Courier", 10))
report_box.pack(pady=10)

# Main training logic
def train_model():
    try:
        train_button.config(state="disabled")
        status_label.config(text="Loading data...", fg="orange")
        root.update_idletasks()

        with open('data.pickle', 'rb') as f:
            dataset = pickle.load(f)

        data = dataset['data']
        labels = dataset['labels']

        if len(data) == 0:
            raise ValueError("No data found in data.pickle")

        sample_label.config(text=f"Samples: {len(data)} | Classes: {len(set(labels))}")
        root.update_idletasks()

        # Split data
        progress_bar["value"] = 10
        root.update_idletasks()
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Train model
        status_label.config(text="Training model...", fg="blue")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Predict
        progress_bar["value"] = 60
        root.update_idletasks()
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=INDEX_TO_LETTER)

        # Save model
        with open('model.pickle', 'wb') as f:
            pickle.dump(model, f)

        progress_bar["value"] = 100
        status_label.config(text=f"✅ Training complete. Accuracy: {acc*100:.2f}%", fg="green")
        report_box.delete(1.0, tk.END)
        report_box.insert(tk.END, f"Accuracy: {acc*100:.2f}%\n\n")
        report_box.insert(tk.END, report)

        messagebox.showinfo("Training Complete", f"Model saved to 'model.pickle'\nAccuracy: {acc*100:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="⚠️ Failed to train model.", fg="red")

    finally:
        train_button.config(state="normal")

# Threaded execution
def run_training_thread():
    Thread(target=train_model, daemon=True).start()

train_button.config(command=run_training_thread)
root.mainloop()