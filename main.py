import face_recognition
import cv2
import numpy as np
import os
import shutil
import pickle
import dlib
from datetime import datetime
import pandas as pd
import time
import tkinter as tk
from tkinter import messagebox
import threading
from PIL import Image, ImageTk

# GPU check
if dlib.DLIB_USE_CUDA:
    print("Using GPU for face recognition")
else:
    print("Warning: Running on CPU. Install CUDA-enabled dlib for GPU acceleration.")

face_data_file = 'face_data.pkl'
known_face_encodings = []
known_face_names = []
dataset_dir = 'dataset/'
dataset_gray = 'dataset_gray/'

def save_face_data():
    with open(face_data_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def load_face_data():
    global known_face_encodings, known_face_names
    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)

def add_face_gui():
    window = tk.Toplevel(root)
    window.title("Add Face")
    window.geometry("300x150")

    tk.Label(window, text="Enter Name:").pack(pady=5)
    name_entry = tk.Entry(window)
    name_entry.pack(pady=5)


    def start_capture():
        name = name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Name cannot be empty")
            return

        add_face_path = os.path.join(dataset_dir, name)
        if os.path.exists(add_face_path):
            messagebox.showerror("Error", "Name already exists")
            return

        os.makedirs(add_face_path)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera could not be opened")
            return
        window.destroy()
        def close_capture():
            cap.release()
            cv2.destroyAllWindows()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            flipped = cv2.flip(frame, 1)
            cv2.imshow("Press 's' to save, 'q' to quit", flipped)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_path = os.path.join(add_face_path, f"{name}_{len(os.listdir(add_face_path)) + 1}.jpg")
                cv2.imwrite(img_path, flipped)
                print(f"Saved: {img_path}")

            if cv2.getWindowProperty("Press 's' to save, 'q' to quit", cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        close_capture()
        process_images()
        load_known_faces()
        save_face_data()
        root.after(0, lambda: messagebox.showinfo("Done", "Face added successfully"))
        root.after(0, window.destroy)

    def start_capture_thread():
        threading.Thread(target=start_capture, daemon=True).start()

    tk.Button(window, text="Start Camera", command=start_capture_thread).pack(pady=10)
    tk.Button(window, text="Close", command=window.destroy).pack()

    

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    hour = now.hour
    time_block = f"{hour:02d}:00-{hour+1:02d}:00"
    
    attendance_dir = 'attendance'
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
    
    file_path = os.path.join(attendance_dir, f"{date_str}.csv")

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Time Block", "Timestamp"])
        df.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    already_marked = ((df["Name"] == name) & (df["Time Block"] == time_block)).any()

    if not already_marked:
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        new_entry = pd.DataFrame([[name, time_block, timestamp]], columns=["Name", "Time Block", "Timestamp"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(file_path, index=False)
        print(f"✅ Attendance marked for {name} for period {time_block}")
    else:
        print(f"Attendance already marked for {name} for period {time_block}")

def process_images():
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        person_gray_dir = os.path.join(dataset_gray, person_name)

        if not os.path.exists(person_gray_dir):
            os.makedirs(person_gray_dir)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            gray_image_path = os.path.join(person_gray_dir, image_name)

            image = cv2.imread(image_path)
            enhanced_image = enhance_contrast(image)
            
            face_locations = face_recognition.face_locations(enhanced_image, model="cnn")  

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = enhanced_image[top:bottom, left:right]
                cv2.imwrite(gray_image_path, face_image)
                print(f"Saved: {gray_image_path}")

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_gray):
        person_dir = os.path.join(dataset_gray, person_name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img, model="cnn")
            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

def recognize_face_gui():
    load_face_data()
    if not known_face_encodings:
        messagebox.showerror("Error", "No known faces found.")
        return

    cap = cv2.VideoCapture(0)
    marked = set()

    def close_window():
        cap.release()
        cv2.destroyAllWindows()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)
        small_frame = cv2.resize(flipped, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            index = np.argmin(distances)
            threshold = 0.5

            name = known_face_names[index] if distances[index] < threshold else "Unknown"

            if name != "Unknown" and name not in marked:
                mark_attendance(name)
                marked.add(name)
                time.sleep(2)

            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(flipped, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(flipped, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Face Recognition (Press q to close)", flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_window()

def delete_face_by_name(name):
    person_dir = os.path.join(dataset_dir, name)
    person_grey_dir = os.path.join(dataset_gray, name)

    for dir_path in [person_dir, person_grey_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as file:
            data = pickle.load(file)
        if isinstance(data, tuple):
            encodings, names = data
            updated_encodings = [enc for enc, n in zip(encodings, names) if n != name]
            updated_names = [n for n in names if n != name]
            with open(face_data_file, 'wb') as f:
                pickle.dump((updated_encodings, updated_names), f)

    load_face_data()
    messagebox.showinfo("Success", f"Deleted face data for {name}")

def delete_face_gui():
    window = tk.Toplevel(root)
    window.title("Delete Face")
    window.geometry("300x150")

    tk.Label(window, text="Enter Name to Delete:").pack(pady=5)
    name_entry = tk.Entry(window)
    name_entry.pack(pady=5)

    def delete_face():
        name = name_entry.get().strip()
        if name:
            threading.Thread(target=delete_face_by_name, args=(name,), daemon=True).start() 
            
        else:
            messagebox.showwarning("Warning", "Please enter a name")

    def delete_face_thread(name):
        delete_face(name)
        # Show info and close window in main thread
        root.after(0, lambda: messagebox.showinfo("Success", f"Deleted face data for {name}"))
        root.after(0, window.destroy)


    tk.Button(window, text="Delete Face", command=delete_face, bg="#f44336", fg="white").pack(pady=10)
    tk.Button(window, text="Close", command=window.destroy).pack()

def recognize_face_gui_thread():
    threading.Thread(target=recognize_face_gui, daemon=True).start()

# GUI Window
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("800x400")  # Reduced height since name-to-delete is removed
root.resizable(False, False)
root.configure(bg='white')

logo_path = "face.jpg"  # update this path
pil_logo = Image.open(logo_path)
pil_logo = pil_logo.resize((400, 400), Image.Resampling.LANCZOS)
logo_img = ImageTk.PhotoImage(pil_logo)

left_frame = tk.Frame(root)
left_frame.pack(side="left", padx=30, pady=30)

tk.Label(left_frame, text="Choose an Option", font=("Arial", 18, "bold")).pack(pady=20)

tk.Button(left_frame, text="Add Face", width=25, height=2, command=add_face_gui,
          bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)

tk.Button(left_frame, text="Recognize Face", width=25, height=2, command=recognize_face_gui_thread,
          bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)

tk.Button(left_frame, text="Delete Face", width=25, height=2, command=delete_face_gui,
          bg="#f44336", fg="white", font=("Arial", 12)).pack(pady=10)

tk.Button(left_frame, text="Exit", width=25, height=2, command=root.destroy,
          bg="#9E9E9E", fg="white", font=("Arial", 12)).pack(pady=10)

right_frame = tk.Frame(root)
right_frame.pack(side="right")

logo_label = tk.Label(right_frame, image=logo_img)
logo_label.pack(side="right")

root.mainloop()