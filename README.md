# Student Performance Predictor

A simple web application that predicts a student's final grade and pass/fail status based on their previous grades and study-related features. Built with **Python**, **TensorFlow**, and **Flask**.

---

## 🛠 Features

- Input student features like `G1`, `G2`, study time, and absences.
- Predict the final grade (`G3`).
- Classify whether the student passes or fails.
- User-friendly web interface with clean, human-like styling.
- Model saved and loaded using Keras.

---

## 📦 Tech Stack

- **Python 3.10**  
- **Flask** – web framework  
- **TensorFlow / Keras** – for the machine learning model  
- **NumPy** and **scikit-learn** – data handling & metrics  
- **HTML/CSS** – frontend styling  

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd student-performance-flask

```markdown
## 2. Create a Python environment (recommended)

**Using conda:**

```bash
conda create -n student_flask python=3.10
conda activate student_flask
```

**Or using venv:**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install flask tensorflow==2.12 numpy==1.24 scikit-learn
```

## 4. Project structure

Make sure your project looks like this:

```
student-flask/
│
├── app.py
├── student_performance_model.keras
└── templates/
    └── index.html
```

- `app.py` → Flask app  
- `student_performance_model.keras` → your trained model  
- `templates/index.html` → HTML form  

## 5. Run the Flask app

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:3000/
```
```
