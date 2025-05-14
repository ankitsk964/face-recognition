import face_recognition
import cv2
import numpy as np
import os
import shutil
import pickle
import dlib

# Check if dlib is using GPU
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

def add_face():
    add_face_name = input("Enter your name: ")
    add_face_path = os.path.join(dataset_dir, add_face_name)

    if os.path.exists(add_face_path):
        print("Name already exists.")
        return

    os.makedirs(add_face_path)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow("Press 's' to capture image, 'q' to quit", flipped_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            image_path = os.path.join(add_face_path, f"{add_face_name}_{len(os.listdir(add_face_path)) + 1}.jpg")
            cv2.imwrite(image_path, flipped_frame)
            print(f"Saved: {image_path}")

        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    process_images()
    load_known_faces()
    save_face_data()

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

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
                print(f"Saved cropped enhanced grayscale image: {gray_image_path}")

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_gray):
        person_dir = os.path.join(dataset_gray, person_name)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            person_image = face_recognition.load_image_file(image_path)

            # Extract multiple encodings
            encodings = face_recognition.face_encodings(person_image, model="cnn")
            
            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

load_known_faces()

def recognize_face():
    if not known_face_encodings:
        print("No known faces. Please add faces first.")
        return

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error capturing frame")
            break

        inverted_frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(inverted_frame, (0, 0), fx=0.25, fy=0.25)

        # Detect faces using CNN model
        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            print(face_distances[best_match_index])
            # **Set a confidence threshold**
            threshold = 0.5  # Adjust based on testing
            if face_distances[best_match_index] < threshold:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            # Scale back the coordinates
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw bounding box
            cv2.rectangle(inverted_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(inverted_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(inverted_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", inverted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def delete_face():
    person_name = input("Enter the name of the person to delete: ")
    person_dir = os.path.join(dataset_dir, person_name)
    person_grey_dir = os.path.join(dataset_gray, person_name)

    # Delete from dataset
    for dir_path in [person_dir, person_grey_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted folder: {dir_path}")
        else:
            print(f"No folder found at: {dir_path}")
    
    # Remove from pickle file
    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as file:
            face_data = pickle.load(file)

        if person_name in face_data:
            del face_data[person_name]
            with open(face_data_file, 'wb') as file:
                pickle.dump(face_data, file)
            print(f"Deleted {person_name}'s data from the pickle file.")
        else:
            print(f"No data found for {person_name} in the pickle file.")
    else:
        print("No pickle file found.")
    
    # Update known face lists
    indices_to_remove = [i for i, name in enumerate(known_face_names) if name == person_name]
    for index in sorted(indices_to_remove, reverse=True):
        del known_face_encodings[index]
        del known_face_names[index]
    
    load_face_data()
    print("Face data updated.")

while True:
    choice = int(input("1. Add Face\n2. Recognize Face\n3. Delete Face\n4. Exit\nChoice: "))
    if choice == 1:
        add_face()
    elif choice == 2:
        load_face_data()
        recognize_face()
    elif choice == 3:
        delete_face()
    else:
        break
