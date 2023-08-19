import React, { useState } from "react";
import RecordRTC from "recordrtc";

const RecordAudio = ({ key }) => {
  const [recording, setRecording] = useState(false);
  const [recorder, setRecorder] = useState(null);
  const [audioData, setAudioData] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const newRecorder = RecordRTC(stream, { type: "audio" });
      setRecorder(newRecorder);
      newRecorder.startRecording();
      setRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = () => {
    if (recorder) {
      recorder.stopRecording(() => {
        setAudioData(recorder.getBlob());
        setRecording(false);
        recorder.reset();
      });
    }
  };

  return (
    <div>
      {!recording ? (
        <button onClick={startRecording}>Start Recording</button>
      ) : (
        <button onClick={stopRecording}>Stop Recording</button>
      )}
      {audioData && (
        <audio controls>
          <source src={URL.createObjectURL(audioData)} />
        </audio>
      )}
    </div>
  );
};

export default RecordAudio;
