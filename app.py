from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import PyPDF2

# Flask App Initialization
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Define the upload directory and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF
def extract_text_from_pdf(filename):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# LLAMA2 LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files.get('file')
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pdf_text = extract_text_from_pdf(filename)
        return jsonify({'pdf_text': pdf_text})
    return jsonify({'error': 'Invalid file format'})

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    pdf_text = request.form.get('pdf_text')
    if input_text and pdf_text:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are a helpful Assistant. You provide detailed and accurate answers based on the following PDF content: {pdf_text}"),
                ("user", f"Question: {input_text}")
            ]
        )
        chain = prompt | llm | output_parser
        response = chain.invoke({"question": input_text})
    else:
        response = "No input provided or PDF not uploaded."
    return jsonify({'response': response})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host="0.0.0.0")
