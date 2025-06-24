from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

# Initialize the face cascade and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# File path to the attendance CSV
attendance_file = 'D:/New projects/Attendance/StudentDetails/studentdetails.csv'

# Student folders (assumes folder names are the student names)
train_images_path = 'train_images'
student_names = {}

# Check if the path exists and has folders
if os.path.exists(train_images_path):
    student_folders = os.listdir(train_images_path)
    for index, folder in enumerate(student_folders):
        student_names[index] = folder
else:
    print(f"Error: {train_images_path} not found!")

# Training data preparation
faces = []
labels = []
for label, student_name in student_names.items():
    student_folder = os.path.join(train_images_path, student_name)
    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label)

# Load the trained recognizer model
recognizer.read('student_recognizer.yml')

# Load label mappings (generated during training)
if os.path.exists('label_map.npy'):
    student_names = np.load('label_map.npy', allow_pickle=True).item()
else:
    print("label_map.npy not found! Make sure to train the model first.")


# Path to the attendance file
attendance_file = r'D:\New projects\Attendance\StudentDetails\studentdetails.csv'

# Function to mark attendance with re-indexed serial numbers and header check
def mark_attendance(student_name):
    # Ensure the attendance file exists and is writable
    if os.access(attendance_file, os.W_OK):
        existing_data = []
        
        # Check if the file is empty or needs a header
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            with open(attendance_file, 'r', newline='') as file:
                reader = csv.reader(file)
                existing_data = [row for row in reader if any(row)]  # Remove blank rows
        
        # If the file is empty or header is missing, add header
        if not existing_data or existing_data[0] != ['Serial No.', 'Enrollment', 'Present or Not', 'Time', 'Name']:
            existing_data = [['Serial No.', 'Enrollment', 'Present or Not', 'Time', 'Name']]

        serial_number = len(existing_data) - 1  # Subtract 1 to exclude header row

        # Check if the student already exists in the data
        student_exists = False
        for row in existing_data[1:]:
            if row[1] == f"1_{student_name}":  # Check if already marked
                student_exists = True
                break
        
        if not student_exists:
            # Add attendance if not already marked, with Name at the end
            existing_data.append([serial_number + 1, f"1_{student_name}", 'Present', datetime.now().strftime('%d-%m-%Y %H:%M'), student_name])
        
        # Re-index serial numbers
        for i, row in enumerate(existing_data[1:], start=1):
            row[0] = i  # Reassign serial numbers
        
        # Write updated data back to the file
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)

        print(f"Attendance marked for {student_name}.")
    else:
        print(f"Error: Cannot write to {attendance_file}. Check file permissions.")

#n = int(input("enter camera index: "))
# Capture video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    recognized_students = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f"Time: {current_time}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 60:
                student_name = student_names.get(label, "Unknown")
                label_text = f"{student_name} ({int(confidence)}%)"
                color = (0, 255, 0)

                if student_name != "Unknown" and student_name not in recognized_students:
                    mark_attendance(student_name)
                    recognized_students.add(student_name)
            else:
                student_name = "Unknown"
                label_text = f"{student_name} ({int(confidence)}%)"
                color = (0, 0, 255)

            # Draw label and rectangle
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('v.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
