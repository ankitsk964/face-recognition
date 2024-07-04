import face_recognition
import cv2
import numpy as np
import os
import shutil
import pickle


face_data_file = 'face_data.pkl'


known_face_encodings = []
known_face_names = []
add_face_name = " "
add_face_path = " "

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
    global add_face_path, add_face_name

    add_face_path = " "
    add_face_name = " "

    def take_input():
        global add_face_path, add_face_name

        add_face_name = str(input("Enter your name: "))
        add_face_path = "dataset/" + add_face_name

        is_exist = os.path.exists(add_face_path)

        if is_exist:
            print("Name already Taken")
            take_input()
        else:
            os.makedirs(add_face_path)

    take_input()
    video_capture_add = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture_add.read()
        invert = cv2.flip(frame, 1)
        cv2.imshow("Video - Press 's' to capture image, 'q' to quit ", invert)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            image_path = os.path.join(add_face_path, f"{add_face_name}_{len(os.listdir(add_face_path)) + 1}.jpg")
            cv2.imwrite(image_path, invert)
            print(f"Captured and saved image to {image_path}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gray_scale()
    video_capture_add.release()
    cv2.destroyAllWindows()


def gray_scale():
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        if os.path.isdir(person_dir):
            person_gray_dir = os.path.join(dataset_gray, person_name)
            if not os.path.exists(person_gray_dir):
                os.makedirs(person_gray_dir)

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                gray_image_path = os.path.join(person_gray_dir, image_name)

                # Read the image
                image = cv2.imread(image_path)

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                face_locations = face_recognition.face_locations(gray_image)

                if face_locations:
                    # Assuming the first detected face is the one we want to crop
                    top, right, bottom, left = face_locations[0]

                    # Crop the face from the grayscale image
                    face_image = gray_image[top:bottom, left:right]

                    # resized_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
                    print("Face cropped from image")
                    cv2.imwrite(gray_image_path, face_image)
                    print(f"Saved cropped grayscale image to {gray_image_path}")
                else:
                    print("No face found")
                # Save the grayscale image


gray_scale()


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_gray):
        person_dir = os.path.join(dataset_gray, person_name)

        if os.path.isdir(person_dir):

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                person_image = face_recognition.load_image_file(image_path)

                face_encoding = face_recognition.face_encodings(person_image)

                if face_encoding:
                    first_face_encoding = face_encoding[0]
                    known_face_encodings.append(first_face_encoding)
                    known_face_names.append(person_name)


load_face_data()


def delete_face():
    person_name = input("Enter the name of the person to delete: ")
    person_dir = os.path.join(dataset_dir, person_name)

    if os.path.exists(person_dir):
        confirmation = input(f"Are you sure you want to delete all data for {person_name}? (y/n): ")
        if confirmation.lower() == 'y':
            shutil.rmtree(person_dir)
            print(f"Data for {person_name} has been deleted.")
            indices_to_remove = [i for i, name in enumerate(known_face_names) if name == person_name]
            for index in sorted(indices_to_remove, reverse=True):
                del known_face_encodings[index]
                del known_face_names[index]
            if os.path.exists('face_data.pkl'):
                with open('face_data.pkl', 'rb') as file:
                    face_data = pickle.load(file)

                if person_name in face_data:
                    del face_data[person_name]

                with open('face_data.pkl', 'wb') as file:
                    pickle.dump(face_data, file)

                print(f"Deleted {person_name}'s data from the pickle file.")
                load_face_data()
            else:
                print("No pickle file found.")
        else:
            print("Deletion canceled.")
    else:
        print(f"No data found for {person_name}.")


def recognize_face():
    if not known_face_encodings:
        print("No known faces \n Please add some known faces")
        return None
    video_capture = cv2.VideoCapture(0)

    while True:

        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame from the video source.")
            break
        inverted_video = cv2.flip(frame, 1)

        small_frame = cv2.resize(inverted_video, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(inverted_video, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(inverted_video, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(inverted_video, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Video", inverted_video)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("releasing camera")
    video_capture.release()
    cv2.destroyAllWindows()


while True:
    choice = int(input(
        "1. To scan a face \n"
        "2. To Recognize a face \n"
        "3. To delete a face \n"
        "4. Exit \n"))

    if choice == 1:
        add_face()
        print("Loading Faces \n")
        load_known_faces()
        save_face_data()
    elif choice == 2:
        load_face_data()
        recognize_face()

    elif choice == 3:
        delete_face()

    else:
        exit(0)
