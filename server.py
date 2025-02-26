from flask import Flask, request, send_file
from flask_cors import CORS
import os
import subprocess  # To run the model script
import sys

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

SAMPLES_FOLDER = "Samples"
INPUT_FOLDER = os.path.join(SAMPLES_FOLDER, "Sample_Test_Input")
OUTPUT_FOLDER = SAMPLES_FOLDER  # Output will be in the Samples folder
os.makedirs(INPUT_FOLDER, exist_ok=True)

@app.route("/denoise", methods=["POST"])
def denoise():
    if "audio" not in request.files:
        return {"error": "No file provided"}, 400

    audio_file = request.files["audio"]
    input_path = os.path.join(INPUT_FOLDER, audio_file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, "denoised.wav")

    audio_file.save(input_path)
    
    # Run the whole MODEL.py script with input and output paths
    try:
        subprocess.run([sys.executable, "MODEL.py"], check=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Model processing failed: {e}"}, 500

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
