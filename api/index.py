import os
import base64
import re
from flask import Flask, request, jsonify, Response
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# Initialize Mistral client
client = Mistral(api_key=MISTRAL_API_KEY)

# Check if file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Encode file to base64 for OCR
def encode_file(file_content, file_ext):
    base64_content = base64.b64encode(file_content).decode("utf-8")
    if file_ext in {"png", "jpg", "jpeg"}:
        return f"data:image/{file_ext};base64,{base64_content}"
    elif file_ext == "pdf":
        return f"data:application/pdf;base64,{base64_content}"
    return None

# Clean extracted text
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove Markdown images
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # Remove headers
    text = re.sub(r"[\*]{1,2}|_{1,2}", "", text)  # Remove bold/italic
    text = re.sub(r"\s*/\s*", " ", text)  # Remove slashes
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

@app.route("/api/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        file_content = file.read()
        file_ext = file.filename.rsplit(".", 1)[1].lower()
        is_image = file_ext in {"png", "jpg", "jpeg"}

        data_url = encode_file(file_content, file_ext)
        if not data_url:
            return jsonify({"error": "Failed to encode file"}), 500

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url" if is_image else "document_url",
                "image_url" if is_image else "document_url": data_url
            }
        )

        if hasattr(ocr_response, "pages") and ocr_response.pages:
            extracted_text = "\n\n".join(
                page.markdown for page in ocr_response.pages if page.markdown
            )
            cleaned_text = clean_text(extracted_text)
            if not cleaned_text or all(
                text.startswith("![") and text.endswith(")")
                for text in [page.markdown for page in ocr_response.pages]
            ):
                return jsonify({
                    "text": "No readable text found.",
                    "warning": f"OCR failed to extract text from the {'image' if is_image else 'PDF'}",
                    "debug": str(ocr_response)
                })
            return jsonify({"text": cleaned_text})
        else:
            return jsonify({
                "text": "No text extracted",
                "debug": str(ocr_response)
            })

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route("/api/download", methods=["POST"])
def download():
    extracted_text = request.form.get("text", "")
    return Response(
        extracted_text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;filename=extracted_text.txt"}
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)