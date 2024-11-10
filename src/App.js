import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [docId, setDocId] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a PDF file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post("https://c47a-35-223-16-116.ngrok-free.app/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setDocId(response.data.doc_id);
      alert(`File uploaded successfully. Document ID: ${response.data.doc_id}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to upload file.");
    } finally {
      setLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!docId) {
      alert("Please upload a document first.");
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post("https://c47a-35-223-16-116.ngrok-free.app/ask/", {
        doc_id: docId,
        question: question
      });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("Error asking question:", error);
      alert("Failed to retrieve answer.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>PDF Q&A App</h1>
      <div className="upload-section">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Uploading..." : "Upload PDF"}
        </button>
      </div>

      {docId && (
        <div className="question-section">
          <h2>Ask a Question</h2>
          <input
            type="text"
            placeholder="Type your question here"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button onClick={handleAskQuestion} disabled={loading}>
            {loading ? "Fetching Answer..." : "Ask Question"}
          </button>
        </div>
      )}

      {answer && (
        <div className="answer-container">
          <h2>Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
