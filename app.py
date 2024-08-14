import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}


your_model = load_model('/u/alenad_guest/Desktop/finalres/bestresumemodel_trial86.keras')

class_labels = [
    'HR', 'Designer', 'Information-Technology', 'Teacher', 'Advocate', 'Business-Development',
    'Healthcare', 'Fitness', 'Agriculture', 'BPO', 'Sales', 'Consultant', 'Digital-Media',
    'Automobile', 'Chef', 'Finance', 'Apparel', 'Engineering', 'Accountant', 'Construction',
    'Public-Relations', 'Banking', 'Arts', 'Aviation'
]

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
def extract_text_from_docx(file_path): 
    """Extract text from a DOCX file.""" 
    doc = Document(file_path)
    text = "" 
    for para in doc.paragraphs: 
        text += para.text + "\n"
    return text

def preprocess_text(text, tokenizer, max_len):
    """Preprocess the input text for prediction."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analyze the content."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(file_path)
        else:
            return "Unsupported file type", 400
        
        # Load tokenizer 
        tokenizer = Tokenizer()  
        max_len = 1500 
        
        
        preprocessed_text = preprocess_text(resume_text, tokenizer, max_len)
        prediction = your_model.predict(preprocessed_text)
        
        print("Raw prediction output:", prediction)
        
        # Get index of highest probability
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_industry = class_labels[predicted_index]
        
        # Return prediction
        return f"File uploaded and analyzed successfully! Predicted Industry: {predicted_industry}"
    
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run app
    app.run(debug=True)
