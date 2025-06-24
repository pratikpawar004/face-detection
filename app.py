from flask import Flask, render_template, Response, request, redirect, url_for, flash,send_file
from flask import Flask, request, jsonify
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import csv
import detect  # Import functions from detect.py
from groq import Groq


app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Paths for Haar Cascade and training images
HaarCascadePath = "haarcascade_frontalface_default.xml"
TrainImagePath = "D:/New projects/Attendance/train_images/"
TrainLabelPath = "D:/New projects/Attendance/TrainingImageLabel/Trainner.yml"
attendance_file = "D:/New projects/Attendance/StudentDetails/studentdetails.csv"


# Ensure the folders exist
if not os.path.exists(TrainImagePath):
    os.mkdir(TrainImagePath)



# Route for capturing image
# Route for capturing image
@app.route('/capture_image', methods=['POST'])
def capture_image():
    enrollment = request.form['enrollment']
    name = request.form['name']

    if not enrollment or not name:
        return jsonify({"error": "Please provide both Enrollment and Name."}), 400

    # Paths and setup
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HaarCascadePath)
    sampleNum = 0
    directory = f"{TrainImagePath}/{enrollment}_{name}"
    os.makedirs(directory, exist_ok=True)

    # Capture images
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            cv2.imwrite(f"{directory}/{name}_{enrollment}_{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow("Capture Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif sampleNum > 400:
            break
    cam.release()
    cv2.destroyAllWindows()

    # Save student info to CSV
    csv_path = 'D:/New projects/Attendance/StudentDetails/studentdetails.csv'
    with open(csv_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=",")
        writer.writerow([enrollment, name])

    # Return a success response to the frontend
    return jsonify({"success": "Image capture successful!"})



train_images_path = 'train_images/'


@app.route('/train_model', methods=['POST'])
def train_model():
    if not os.path.exists(train_images_path):
        flash("Training images folder does not exist.", "error")
        return redirect(url_for('attendance'))

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = []
        labels = []
        student_names = {}
        label_count = 0

        for folder in os.listdir(train_images_path):
            folder_path = os.path.join(train_images_path, folder)
            if os.path.isdir(folder_path):
                student_names[label_count] = folder

                for image_file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_file)
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)

                    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    for (x, y, w, h) in faces_detected:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (200, 200))
                        faces.append(face)
                        labels.append(label_count)

                label_count += 1

        if faces and labels:
            recognizer.train(faces, np.array(labels))
            recognizer.save('student_recognizer.yml')
            np.save('label_map.npy', student_names)
            flash("Training complete. Model saved successfully!", "success")
        else:
            flash("No faces found. Make sure images are valid face photos.", "error")

    except Exception as e:
        flash(f"Training failed: {str(e)}", "error")

    return redirect(url_for('attendance'))



# Route to handle attendance manually
# Route to handle attendance submission
@app.route('/fill_attendance', methods=['POST'])
def fill_attendance():
    enrollment = request.form.get('enrollment', '').strip()
    name = request.form.get('name', '').strip()
    present_or_not = request.form.get('present_or_not', '').strip()

    if not enrollment or not name or not present_or_not:
        flash("Please provide Enrollment, Name, and Attendance status.", "error")
        return redirect(url_for('attendance'))

    # Get current timestamp
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # Create attendance data dictionary
    attendance_data = {"Enrollment": enrollment, "Present or Not": present_or_not, "Time": time, "Name": name}

    # Check if the attendance file exists
    if os.path.exists(attendance_file):
        # Load existing attendance data
        df = pd.read_csv(attendance_file)

        # Check if the student already exists
        student_exists = df[df['Enrollment'] == enrollment]

        if not student_exists.empty:
            # Update existing student's record
            index = student_exists.index[0]
            df.at[index, "Present or Not"] = present_or_not
            df.at[index, "Time"] = time
            df.at[index, "Name"] = name
        else:
            # Add new student's attendance record
            new_serial_no = len(df) + 1
            new_row = pd.DataFrame([{"Serial No.": new_serial_no, "Enrollment": enrollment, 
                                     "Present or Not": present_or_not, "Time": time, "Name": name}])
            df = pd.concat([df, new_row], ignore_index=True)

        # Update Serial No. column to ensure it's sequential
        df["Serial No."] = range(1, len(df) + 1)

        # Save updated attendance data to the file
        df.to_csv(attendance_file, index=False)
    else:
        # If the file doesn't exist, create it with the new data
        df = pd.DataFrame([{"Serial No.": 1, "Enrollment": enrollment, 
                            "Present or Not": present_or_not, "Time": time, "Name": name}])
        df.to_csv(attendance_file, index=False)

    flash(f"Attendance marked successfully for {name} ({enrollment}).", "success")
    return redirect(url_for('attendance'))

    
TrainImagePath = "D:/New projects/Attendance/train_images/"

    # Route to view the attendance
@app.route('/view_attendance')
def view_attendance():
    # Check if the attendance file exists
    if os.path.exists(attendance_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(attendance_file)

        # Convert the DataFrame to a list of dictionaries
        attendance_data = df.to_dict(orient='records')

        # Render the HTML page with the attendance data
        return render_template('attendance_view.html', attendance_data=attendance_data)
    else:
        return "Attendance file not found.", 400
    




# Define your file path
attendance_file = "StudentDetails/studentdetails.csv"  # Replace with your actual CSV file path

# Initialize the Groq client with your API key
groq_client = Groq(api_key="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")

# Conversation history for each user
user_conversations = {}



# Chat with Llama 3 function
def chat_with_llama(user_id, user_message):
    """Interact with the Llama 3 API."""
    if user_id not in user_conversations:
        user_conversations[user_id] = []  # Initialize conversation history for the user

    # Add user input to the conversation history
    user_conversations[user_id].append({"role": "user", "content": user_message})

    # Call the Llama 3 API
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=user_conversations[user_id],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the assistant's response
        assistant_response = completion.choices[0].message.content

        # Add the response to the conversation history
        user_conversations[user_id].append({"role": "assistant", "content": assistant_response})

        return assistant_response

    except Exception as e:
        return f"An error occurred: {e}"




# Ensure the CSV file exists with proper headers
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Serial No.', 'Enrollment', 'Present or Not', 'Time', 'Name'])



# Route to display all enrolled students
@app.route('/view_enrolled_students')
def view_enrolled_students():
    try:
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            students = [row for row in reader]
    except FileNotFoundError:
        students = []

    return render_template('view_students.html', students=students)


# Route to delete a student by enrollment number
@app.route('/delete_student/<enrollment>')
def delete_student(enrollment):
    try:
        with open(attendance_file, 'r') as file:
            rows = list(csv.reader(file))

        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in rows:
                if row[1] != enrollment:  # Check against Enrollment column
                    writer.writerow(row)
    except FileNotFoundError:
        pass

    return "<script>alert('Student deleted!'); window.location.href='/view_enrolled_students';</script>"


# Route to handle uploaded files
@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory('uploads', filename)


# Add student route (for demonstration purposes)
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        serial_no = request.form['serial_no']
        enrollment = request.form['enrollment']
        present_or_not = request.form['present_or_not']
        time = request.form['time']
        name = request.form['name']

        # Append new student to the CSV file
        with open(attendance_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([serial_no, enrollment, present_or_not, time, name])

        return redirect(url_for('view_enrolled_students'))

    return render_template('add_student.html')  # Create a simple add_student.html form




# Chat route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    """Handle the chat request from the frontend."""
    user_message = request.json.get('message')
    user_id = request.json.get('user_id')

    # Get the chatbot response from Llama 3
    bot_response = chat_with_llama(user_id, user_message)

    # Return the bot's response as JSON
    return jsonify({'reply': bot_response})


@app.route('/clear_all_attendance', methods=['POST'])
def clear_all_attendance():
    try:
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Serial No.', 'Enrollment', 'Present or Not', 'Time', 'Name'])
        flash("All attendance records cleared successfully!", "success")
    except Exception as e:
        flash(f"Error clearing attendance records: {str(e)}", "danger")

    return redirect(url_for('view_enrolled_students'))

   
@app.route('/download_attendance')
def download_attendance():
    attendance_file = "D:/New projects/Attendance/StudentDetails/studentdetails.csv"
    
    # ✅ Correct usage
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Attendance of {today}.csv"
    
    return send_file(attendance_file, as_attachment=True, download_name=filename)





    
@app.route('/')
def index():
    return render_template('index.html')  # Main page

# Route for the attendance page
@app.route('/video')
def video_attendance():
    return render_template('v.html')

@app.route('/attendance')
def attendance():
    return render_template('Attendance.html')  # About page

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Contact page




# Route to stream the video feed from the webcam
@app.route('/video_feed')
def video_feed():
    return Response(detect.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 