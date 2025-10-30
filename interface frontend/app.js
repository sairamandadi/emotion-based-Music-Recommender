import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [emotion, setEmotion] = useState("None");
  const [image, setImage] = useState(null);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => setImage(ev.target.result);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="container">
      <div className="title">
        <span role="img" aria-label="music">🎵</span> Emotion-Based Music Recommender
      </div>
      <div className="section-header">How are you feeling today?</div>
      <div className="flex-row">
        <div className="preview-box">
          {image ? (
            <div
              className="image-preview"
              style={{ backgroundImage: `url('${image}')` }}
            ></div>
          ) : (
            <div className="image-preview" />
          )}
        </div>
        <div className="upload-side">
          <label className="upload-label">
            <span role="img" aria-label="folder">📁</span> Upload Image
          </label>
          <input
            type="file"
            accept="image/*"
            className="file-input"
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          <button
            className="analyze-btn"
            onClick={() => {}}
          >
            <span role="img" aria-label="search">🔍</span> Analyze Emotion
          </button>
        </div>
        <div className="emotion-status" id="emotionStatus">
          Detected Emotion: {emotion}
        </div>
      </div>
      <div className="section-header">Songs for your current mood:</div>
      <div className="song-card">
        <div className="album-cover">
          <img
            id="albumImg"
            src="https://img.icons8.com/ios-filled/50/cccccc/musical-notes.png"
            alt="Album Cover"
            width="40"
            height="40"
          />
        </div>
        <div className="song-details">
          <div className="song-title" id="songTitle">
            Song Title
          </div>
          <div className="artist-name" id="artistName">
            Artist Name
          </div>
        </div>
        <button className="play-btn" id="playBtn">
          <span role="img" aria-label="play">▶</span> Play
        </button>
      </div>
    </div>
  );
}

export default App; 