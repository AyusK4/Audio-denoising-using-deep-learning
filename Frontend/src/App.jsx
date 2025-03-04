import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/src/components/ui/card"
import { Button } from "@/src/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/src/components/ui/tabs"
import { Mic, Upload, AudioWaveformIcon as Waveform, AlertCircle, Loader2 } from "lucide-react"
import { Progress } from "@/src/components/ui/progress"
import { toast } from "sonner"
import { Alert, AlertDescription } from "@/src/components/ui/alert"

export default function SpeechDenoiser() {
  const [audioFile, setAudioFile] = useState(null)
  const [denoisedAudio, setDenoisedAudio] = useState(null)
  const [recording, setRecording] = useState(false)
  const [recordedAudio, setRecordedAudio] = useState(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState("record")
  const [audioName, setAudioName] = useState(null)

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const intervalRef = useRef(null)

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      setAudioFile(file)
      setRecordedAudio(URL.createObjectURL(file))
      setAudioName(file.name)
      setDenoisedAudio(null)
      toast({
        title: "Audio file selected",
        description: `${file.name} (${(file.size / 1024).toFixed(2)} KB)`,
      })
    }
  }

  // Start recording
  const startRecording = async () => {
    try {
      setRecordedAudio(null)
      setDenoisedAudio(null)
      setError(null)
      setLoading(false)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" })
        const file = new File([audioBlob], "recorded_audio.webm", { type: "audio/webm" })
        setAudioFile(file)
        setRecordedAudio(URL.createObjectURL(audioBlob))
        setAudioName("Recorded Audio")
        setRecordingTime(0)
        clearInterval(intervalRef.current)

        toast({
          title: "Recording completed",
          description: `${recordingTime} seconds recorded`,
        })
      }

      mediaRecorderRef.current.start()
      setRecording(true)
      setRecordingTime(0)
      intervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)

      toast({
        title: "Recording started",
        description: "Speak clearly into your microphone",
      })
    } catch (err) {
      setError("Microphone access denied. Please allow microphone access and try again.")
      toast({
        variant: "destructive",
        title: "Recording failed",
        description: "Could not access microphone. Please check permissions.",
      })
    }
  }

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }

      // Stop all audio tracks
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
    }
  }

  // Send audio to backend
  const handleUpload = async () => {
    if (!audioFile) {
      toast({
        variant: "destructive",
        title: "No audio file",
        description: "Please record or select an audio file first",
      })
      return
    }

    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append("audio", audioFile)

    try {
      toast({
        title: "Processing audio",
        description: "This may take a few moments...",
      })

      const response = await fetch("http://localhost:5000/denoise", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`)
      }

      const blob = await response.blob()
      setDenoisedAudio(URL.createObjectURL(blob))
      setLoading(false)

      toast({
        title: "Audio denoised successfully",
        description: "Your audio has been processed and is ready to play",
      })
    } catch (err) {
      console.error(err)
      setLoading(false)
      setError("Failed to process audio. Please try again later.")

      toast({
        variant: "destructive",
        title: "Processing failed",
        description: "Could not denoise audio. Server may be unavailable.",
      })
    }
  }

  // Format time for display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop()
        mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  return (
    <div className="flex items-center justify-center min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
      <Card className="w-full max-w-2xl shadow-xl border-slate-700 bg-slate-800/50 backdrop-blur-sm">
        <CardHeader className="text-center border-b border-slate-700 pb-6">
          <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
            Audio Denoiser
          </CardTitle>
          <CardDescription className="text-slate-300 mt-2">
            Remove background noise from your audio recordings
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-6 pb-2">
          <Tabs defaultValue="record" value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-2 mb-6">
              <TabsTrigger value="record" disabled={recording || loading}>
                <Mic className="mr-2 h-4 w-4" />
                Record Audio
              </TabsTrigger>
              <TabsTrigger value="upload" disabled={recording || loading}>
                <Upload className="mr-2 h-4 w-4" />
                Upload File
              </TabsTrigger>
            </TabsList>

            <TabsContent value="record" className="space-y-4">
              <div className="flex justify-center">
                <Button
                  onClick={recording ? stopRecording : startRecording}
                  className={`w-40 h-40 rounded-full flex flex-col items-center justify-center gap-2 transition-all ${
                    recording ? "bg-red-500 hover:bg-red-600 animate-pulse" : "bg-blue-500 hover:bg-blue-600"
                  }`}
                >
                  <Mic className={`h-10 w-10 ${recording ? "animate-bounce" : ""}`} />
                  <span className="font-medium">{recording ? "Stop" : "Start"}</span>
                  {recording && <span className="text-xs">{formatTime(recordingTime)}</span>}
                </Button>
              </div>

              {recording && (
                <div className="flex items-center justify-center gap-2 mt-4">
                  <Waveform className="h-5 w-5 text-red-400 animate-pulse" />
                  <span className="text-red-400 animate-pulse">Recording in progress...</span>
                </div>
              )}
            </TabsContent>

            <TabsContent value="upload">
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                <Upload className="h-10 w-10 text-slate-400 mb-4" />
                <p className="text-slate-300 mb-4">Drag and drop your audio file here or click to browse</p>
                <Button variant="outline" className="relative">
                  Select Audio File
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                </Button>
              </div>
            </TabsContent>
          </Tabs>

          {error && (
            <Alert variant="destructive" className="mt-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {recordedAudio && (
            <div key={recordedAudio} className="mt-6 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-slate-300">Original Audio</h3>
                <span className="text-xs text-slate-400">{audioName}</span>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <audio controls className="w-full">
                  <source src={recordedAudio} />
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
          )}

          {loading && (
            <div className="mt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                <p className="text-sm text-slate-300">Processing your audio...</p>
              </div>
              <Progress value={45} className="h-2" />
            </div>
          )}

          {denoisedAudio && (
            <div key={denoisedAudio} className="mt-6 space-y-2">
              <h3 className="text-sm font-medium text-slate-300">Denoised Audio</h3>
              <div className="bg-slate-700/50 rounded-lg p-3 border border-green-500/20">
                <audio controls className="w-full">
                  <source src={denoisedAudio} />
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-center pt-2 pb-6">
          <Button
            onClick={handleUpload}
            disabled={!audioFile || loading}
            className="w-full max-w-xs"
            variant={loading ? "outline" : "default"}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Waveform className="mr-2 h-4 w-4" />
                Denoise Audio
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}


