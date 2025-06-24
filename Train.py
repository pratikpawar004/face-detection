import cv2
import os
import numpy as np

# Path to the training images folder
train_images_path = 'train_images/'

# Initialize the face recognizer and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
faces = []
labels = []
student_names = {}
label_count = 0

# Iterate over student folders
for student_folder in os.listdir(train_images_path):
    folder_path = os.path.join(train_images_path, student_folder)

    if os.path.isdir(folder_path):
        student_names[label_count] = student_folder  # Map label to folder name

        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, image_file)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Could not read image: {image_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_detected:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))  # Optional: standardize size
                    faces.append(face)
                    labels.append(label_count)

        label_count += 1

# Train the recognizer
if faces and labels:
    recognizer.train(faces, np.array(labels))
    recognizer.save('student_recognizer.yml')
    np.save('label_map.npy', student_names)
    print("✅ Training complete. Model and label map saved.")
else:
    print("❌ No faces found for training. Check your images.")
