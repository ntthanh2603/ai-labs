/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect, useRef } from "react";
import "./App.css";

// Simplified declarations for browser SpeechRecognition
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [language, setLanguage] = useState("vi-VN");
  const [isCopied, setIsCopied] = useState(false);
  const [isSupported] = useState(() => {
    return !!(
      typeof window !== "undefined" &&
      (window.SpeechRecognition || window.webkitSpeechRecognition)
    );
  });
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!isSupported) return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;

    recognition.onresult = (event: any) => {
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcriptPiece = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcriptPiece + " ";
        }
      }

      setTranscript((prev) => prev + finalTranscript);
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error);
      setError(`Lỗi: ${event.error}`);
      setIsRecording(false);
    };

    recognition.onend = () => {
      if (isRecording) {
        recognition.start();
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [language, isRecording, isSupported]);

  const toggleRecording = () => {
    setError("");

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current?.start();
        setIsRecording(true);
      } catch (err) {
        setError("Không thể bắt đầu ghi âm. Vui lòng thử lại.");
        console.error(err);
      }
    }
  };

  const clearTranscript = () => {
    setTranscript("");
    setError("");
  };

  const copyToClipboard = () => {
    if (transcript) {
      navigator.clipboard
        .writeText(transcript)
        .then(() => {
          setIsCopied(true);
          setTimeout(() => setIsCopied(false), 2000);
        })
        .catch((err) => setError("Không thể sao chép: " + err));
    }
  };

  const downloadTranscript = () => {
    if (!transcript) return;
    const element = document.createElement("a");
    const file = new Blob([transcript], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `transcript-${new Date().getTime()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎤 Speech to Text</h1>
        <p className="subtitle">Ghi âm và chuyển đổi giọng nói thành văn bản</p>
      </header>

      <div className="container">
        <div className="card">
          <div className="record-section">
            <button
              className={`record-button ${isRecording ? "recording" : ""}`}
              onClick={toggleRecording}
              aria-label={isRecording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
            >
              {isRecording ? "⏸️" : "🎤"}
            </button>

            <div className={`status ${isRecording ? "recording" : ""}`}>
              <div className="status-main">
                <span className="status-indicator"></span>
                <span>{isRecording ? "Đang lắng nghe..." : "Sẵn sàng"}</span>
              </div>
              {isRecording && (
                <div className="waveform">
                  <div className="wave-bar"></div>
                  <div className="wave-bar"></div>
                  <div className="wave-bar"></div>
                  <div className="wave-bar"></div>
                  <div className="wave-bar"></div>
                </div>
              )}
            </div>

            <div className="language-selector">
              <label htmlFor="language">Ngôn ngữ:</label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isRecording}
              >
                <option value="vi-VN">Tiếng Việt</option>
                <option value="en-US">English (US)</option>
                <option value="en-GB">English (UK)</option>
                <option value="ja-JP">日本語</option>
                <option value="ko-KR">한국어</option>
                <option value="zh-CN">中文 (简体)</option>
                <option value="fr-FR">Français</option>
                <option value="de-DE">Deutsch</option>
                <option value="es-ES">Español</option>
              </select>
            </div>
          </div>
        </div>

        <div className="card transcript-section">
          <div className="transcript-label">📝 Văn bản nhận diện</div>
          <div className="transcript-content">{transcript}</div>

          {!isSupported && (
            <div className="error">
              ⚠️ Trình duyệt của bạn không hỗ trợ Speech Recognition. Vui lòng
              sử dụng Chrome hoặc Edge.
            </div>
          )}

          {transcript && (
            <div className="controls">
              <button className="btn btn-success" onClick={copyToClipboard}>
                {isCopied ? "✓ Đã chép" : "� Sao chép"}
              </button>
              <button className="btn btn-primary" onClick={downloadTranscript}>
                📥 Tải xuống
              </button>
              <button className="btn btn-secondary" onClick={clearTranscript}>
                🗑 Xóa nội dung
              </button>
            </div>
          )}

          {error && <div className="error">⚠️ {error}</div>}
        </div>
      </div>
    </div>
  );
}

export default App;
