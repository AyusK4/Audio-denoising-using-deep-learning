from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import subprocess
import sys
import ffmpeg 

app = Flask(__name__)
CORS(app)

SAMPLES_FOLDER = "Samples"
INPUT_FOLDER = os.path.join(SAMPLES_FOLDER, "Sample_Test_Input")
OUTPUT_FOLDER = SAMPLES_FOLDER  # Output will be in the Samples folder
os.makedirs(INPUT_FOLDER, exist_ok=True)

def convert_to_wav(input_path, output_path):
    """ Convert any browser-recorded file (WebM/OGG) to WAV using ffmpeg. """
    try:
        ffmpeg.input(input_path).output(output_path, format="wav", acodec="pcm_s16le").run(overwrite_output=True)
        return True
    except Exception as e:
        print("Error converting to WAV:", e)
        return False

@app.route("/denoise", methods=["POST"])
def denoise():
    if "audio" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["audio"]
    input_path = os.path.join(INPUT_FOLDER, audio_file.filename)
    wav_path = os.path.join(INPUT_FOLDER, "converted_input.wav")  # Ensuring WAV format
    output_path = os.path.join(OUTPUT_FOLDER, "denoised.wav")

    audio_file.save(input_path)

    # Convert WebM/OGG/M4A to WAV
    if not audio_file.filename.endswith(".wav"):
        success = convert_to_wav(input_path, wav_path)
        if not success:
            return jsonify({"error": "Audio conversion failed"}), 500
    else:
        wav_path = input_path  # If already WAV, use original file

    # Run MODEL.py with the converted WAV file
    try:
        subprocess.run([sys.executable, "MODEL.py"], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Model processing failed: {e}"}), 500

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
