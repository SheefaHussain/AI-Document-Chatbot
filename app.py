import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PyPDF2 import PdfReader
import docx
import io
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

uploaded_document_text = ""

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        reader = PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Could not extract text from PDF: {e}")
    return text

def extract_text_from_docx(file_stream):
    text = ""
    try:
        document = docx.Document(file_stream)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        raise ValueError(f"Could not extract text from DOCX: {e}")
    return text

def extract_text_from_txt(file_stream):
    try:
        return file_stream.read().decode('utf-8')
    except UnicodeDecodeError:
        file_stream.seek(0)
        return file_stream.read().decode('iso-8859-1')
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        raise ValueError(f"Could not extract text from TXT: {e}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set. API calls may fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured.")

@app.route('/')
def home():
    return "Hello from Flask! Your chatbot backend is running."

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global uploaded_document_text

    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_stream = io.BytesIO(file.read())

        extracted_text = ""
        try:
            if file_extension == '.pdf':
                extracted_text = extract_text_from_pdf(file_stream)
            elif file_extension == '.docx':
                extracted_text = extract_text_from_docx(file_stream)
            elif file_extension == '.txt' or file_extension == '.md':
                extracted_text = extract_text_from_txt(file_stream)
            else:
                return jsonify({"message": "Unsupported file type"}), 400

            uploaded_document_text = extracted_text
            # --- MODIFIED LINE BELOW ---
            # Now returning the extracted_text directly, as the frontend expects it.
            return jsonify({"message": "File processed successfully", "extracted_text": extracted_text, "extracted_text_length": len(extracted_text)}), 200

        except ValueError as e:
            return jsonify({"message": str(e)}), 400
        except Exception as e:
            print(f"An unexpected error occurred during file processing: {e}")
            return jsonify({"message": f"An unexpected error occurred: {e}"}), 500

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    global uploaded_document_text

    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"message": "No prompt provided"}), 400

    if not uploaded_document_text:
        return jsonify({"message": "No document uploaded yet. Please upload a file first using /upload_file."}), 400

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"message": "Gemini API key not set in environment variables."}), 500

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        full_prompt_content = [
            {"text": f"Based on the following document, answer the user's question:\n\nDocument:\n{uploaded_document_text}\n\nUser's question: {prompt}"}
        ]

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": full_prompt_content
                }
            ]
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            ai_response_text = result['candidates'][0]['content']['parts'][0].get('text', '')
            return jsonify({"ai_response": ai_response_text}), 200
        else:
            print("Unexpected response structure from Gemini API:", result)
            return jsonify({"ai_response": "Could not get a valid response from the AI model."}), 500

    except requests.exceptions.RequestException as req_err:
        print(f"Error calling Gemini API (requests error): {req_err}")
        return jsonify({"message": f"Error interacting with AI (network/API call issue): {req_err}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred in ask_gemini: {e}")
        return jsonify({"message": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
