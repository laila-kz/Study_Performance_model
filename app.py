from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model with error handling - try both .h5 and .keras formats
model = None
try:
    model = load_model("student_performance_model.h5", compile=False)
    print("Model loaded successfully from .h5 file!")
except Exception as e:
    print(f"Failed to load .h5 model: {e}")
    try:
        model = load_model("student_performance_model.keras", compile=False)
        print("Model loaded successfully from .keras file!")
    except Exception as e2:
        print(f"Failed to load .keras model: {e2}")
        print("Error: Could not load model from either format.")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None:
                return render_template("index.html", grade=None, result=None, 
                                     error="Model not loaded. Please check the model file.")
            
            # Collect and validate form data
            study_time = float(request.form['study_time'])
            absences = float(request.form['absences'])
            g1 = float(request.form['G1'])
            g2 = float(request.form['G2'])
            
            # Input validation
            if study_time < 0 or absences < 0:
                return render_template("index.html", grade=None, result=None,
                                     error="Study time and absences must be non-negative.")
            
            if not (0 <= g1 <= 20) or not (0 <= g2 <= 20):
                return render_template("index.html", grade=None, result=None,
                                     error="Grades must be between 0 and 20.")
            
            # Create features array in correct order
            features = np.array([[study_time, absences, g1, g2]])
            
            # Make prediction
            pred = model.predict(features, verbose=0)
            grade = float(pred[0][0])
            result = "Pass" if grade >= 10 else "Fail"
            
            return render_template("index.html", grade=grade, result=result)
            
        except ValueError as e:
            return render_template("index.html", grade=None, result=None, 
                                 error="Please enter valid numbers for all fields.")
        except Exception as e:
            return render_template("index.html", grade=None, result=None, 
                                 error=f"An error occurred: {str(e)}")
    
    return render_template("index.html", grade=None, result=None)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
