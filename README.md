# Face Detection Attendance System

This is a Flask-based Face Detection Attendance System using OpenCV. It allows capturing student faces, training a model, and marking attendance based on facial recognition.

---

## 🚀 Features

- Real-time face detection using webcam
- Face dataset collection
- Model training using LBPH recognizer
- Attendance marking via face recognition
- Attendance report in CSV format
- Web-based interface
- Integration with Groq-powered LLaMA 3 chatbot (optional)

---

## 🛠️ Software Requirements

- **Python 3.8+**
- **pip** (Python package manager)
- **OpenCV**
- **Flask**
- **NumPy**
- **Pandas**
- **Pillow (PIL)**
- **Groq SDK** *(optional, for chatbot integration)*

---


## 🔧 Installation & Setup

1. **Clone the repo**
    ```bash
    git clone https://github.com/pratikpawar004/face-detection.git
    cd face-detection
    ```

2. **Set up virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate       # macOS/Linux
    .\venv\Scripts\activate        # Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**
    ```bash
    python app.py
    ```
    Then visit: `http://127.0.0.1:5000/`

---

## 🚀 Usage

- **Capture images**: `http://127.0.0.1:5000/capture_image`
- **Train model**: `http://127.0.0.1:5000/train_model`
- **Fill attendance**: `http://127.0.0.1:5000/fill_attendance`
- **Download report**: `attendance.csv`

---

## 📂 Project Structure


