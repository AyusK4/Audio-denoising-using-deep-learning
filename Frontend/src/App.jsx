import React, { useState, useRef } from "react";

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [denoisedAudio, setDenoisedAudio] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const intervalRef = useRef(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
      setAudioFile(file);
      setRecordedAudio(URL.createObjectURL(file));
    }
  };

  // Start recording
  const startRecording = async () => {
    setRecordedAudio(null);
    setError(false);
    setLoading(false);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunksRef.current = [];

    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      setAudioFile(audioBlob);
      setRecordedAudio(URL.createObjectURL(audioBlob));
      setRecordingTime(0);
      clearInterval(intervalRef.current);
    };

    mediaRecorderRef.current.start();
    setRecording(true);
    setRecordingTime(0);
    intervalRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  };

  // Stop recording
  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
    clearInterval(intervalRef.current);
  };

  // Send audio to backend
  const handleUpload = async () => {
    if (!audioFile) return alert("No audio file selected!");
    setLoading(true);
    setError(false);
    const formData = new FormData();
    formData.append("audio", audioFile);
    try {
      const response = await fetch("http://localhost:5000/denoise", {
        method: "POST",
        body: formData,
      });
      const blob = await response.blob();
      setDenoisedAudio(URL.createObjectURL(blob));
      setLoading(false);
      setError(false);
    } catch (error) {
      console.log(error);
      setLoading(false);
      setError(true);
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-800">
      <div className="p-6  mx-auto bg-gray-300 rounded-xl shadow-md space-y-4">
        <h2 className="text-xl font-bold">Speech Denoiser</h2>

        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="border p-2 w-full"
        />
        <div className="flex items-center justify-between">
          <button
            onClick={recording ? stopRecording : startRecording}
            className="bg-blue-500 text-white p-2 rounded"
          >
            {recording ? "Stop Recording" : "Start Recording"}
          </button>

          <button
            onClick={handleUpload}
            disabled={loading}
            className="bg-green-500 text-white p-2 rounded"
          >
            Upload & Denoise
          </button>
        </div>
        {recording && (
          <p className="text-red-500">Recording... {recordingTime}s</p>
        )}
        {recordedAudio && (
          <div>
            <p className="text-gray-700">Original Audio:</p>
            <audio controls>
              <source src={recordedAudio} type="audio/wav" />
            </audio>
          </div>
        )}
        {loading && <div>Loading ...</div>}
        {error && <div className="text-red-700">Error: Try again later</div>}
        {denoisedAudio && (
          <div>
            <p className="text-gray-700">Denoised Audio:</p>
            <audio controls>
              <source src={denoisedAudio} type="audio/wav" />
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
