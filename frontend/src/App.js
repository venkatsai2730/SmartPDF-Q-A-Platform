import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = "http://localhost:8000";

/* ================================================================
   LOGIN PAGE COMPONENT
   ================================================================ */
function LoginPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isRegister) {
        const res = await axios.post(`${API_BASE}/register`, {
          username,
          password,
          full_name: fullName,
        });
        onLogin(res.data);
      } else {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const res = await axios.post(`${API_BASE}/login`, formData, {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });
        onLogin(res.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-bg-shape login-bg-1" />
      <div className="login-bg-shape login-bg-2" />

      <div className="login-card">
        <div className="login-logo">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 18V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M9 15L12 12L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <h1 className="login-title">SmartPDF Q&A</h1>
        <p className="login-subtitle">AI-Powered Document Intelligence</p>

        <form onSubmit={handleSubmit} className="login-form">
          {isRegister && (
            <div className="form-group">
              <label htmlFor="fullname-input">Full Name</label>
              <input
                id="fullname-input"
                type="text"
                placeholder="Enter your full name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>
          )}
          <div className="form-group">
            <label htmlFor="username-input">Username</label>
            <input
              id="username-input"
              type="text"
              placeholder="Enter username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password-input">Password</label>
            <input
              id="password-input"
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && <div className="login-error">{error}</div>}

          <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
            {loading ? (
              <><span className="spinner" /> {isRegister ? 'Creating Account…' : 'Signing In…'}</>
            ) : (
              isRegister ? 'Create Account' : 'Sign In'
            )}
          </button>
        </form>

        <p className="login-switch">
          {isRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button className="link-btn" onClick={() => { setIsRegister(!isRegister); setError(''); }}>
            {isRegister ? 'Sign In' : 'Create Account'}
          </button>
        </p>

        <div className="login-demo-hint">
          <strong>Demo:</strong> username <code>admin</code> / password <code>admin123</code>
        </div>
      </div>
    </div>
  );
}


/* ================================================================
   MAIN APP COMPONENT (after login)
   ================================================================ */
function MainApp({ token, user, onLogout }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [documents, setDocuments] = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [question, setQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [askError, setAskError] = useState('');
  const [showSources, setShowSources] = useState(false);

  const fileInputRef = useRef(null);

  const authHeaders = { Authorization: `Bearer ${token}` };

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setFile(selectedFile);
      setUploadError('');
    } else if (selectedFile) {
      setUploadError('Only PDF files are supported.');
      setFile(null);
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files[0]);
  }, []);

  const handleDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true); }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleUpload = async () => {
    if (!file) { setUploadError('Please select a PDF file first.'); return; }

    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
      setUploadError('File too large. Maximum size is 50MB.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploading(true);
      setUploadError('');
      // NOTE: Do NOT set Content-Type manually for FormData — axios/browser
      // must set it automatically to include the multipart boundary.
      const response = await axios.post(`${API_BASE}/upload/`, formData, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 300000, // 5 min timeout for large PDFs
      });
      const newDoc = {
        doc_id: response.data.doc_id,
        filename: response.data.filename,
        num_chunks: response.data.num_chunks,
        num_pages: response.data.num_pages,
        embedding_time_ms: response.data.embedding_time_ms,
      };
      setDocuments((prev) => [...prev, newDoc]);
      setActiveDocId(response.data.doc_id);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        setUploadError('Upload timed out. The file may be too large or the server is busy.');
      } else if (error.response?.status === 503) {
        setUploadError('Models are still loading. Please wait a moment and try again.');
      } else {
        const detail = error.response?.data?.detail || 'Upload failed. Is the backend running?';
        setUploadError(detail);
      }
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!activeDocId) { setAskError('Please upload and select a document first.'); return; }
    if (!question.trim()) { setAskError('Please enter a question.'); return; }

    try {
      setAsking(true);
      setAskError('');
      setAnswer(null);
      const response = await axios.post(`${API_BASE}/ask/`, {
        doc_id: activeDocId, question,
      }, {
        headers: authHeaders,
        timeout: 120000, // 2 min timeout for QA pipeline
      });
      setAnswer(response.data);
    } catch (error) {
      if (error.response?.status === 401) {
        setAskError('Session expired. Please log out and sign in again.');
      } else if (error.response?.status === 503) {
        setAskError('Models are still loading. Please wait a moment and try again.');
      } else {
        setAskError(error.response?.data?.detail || 'Query failed. Is the backend running?');
      }
    } finally {
      setAsking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  };

  const activeDoc = documents.find((d) => d.doc_id === activeDocId);
  const confidencePercent = answer ? (answer.score * 100).toFixed(1) : 0;

  return (
    <div className="app-wrapper">
      <div className="bg-shape bg-shape-1" />
      <div className="bg-shape bg-shape-2" />

      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="header-top">
            <div className="logo-section">
              <div className="logo-icon">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div>
                <h1>SmartPDF Q&A</h1>
                <p className="header-subtitle">AI-Powered Document Intelligence</p>
              </div>
            </div>
            <div className="user-section">
              <span className="user-name">👋 {user.full_name}</span>
              <button className="btn btn-outline btn-sm" onClick={onLogout} id="logout-btn">Logout</button>
            </div>
          </div>
        </header>

        <div className="main-grid">
          {/* Left Panel */}
          <div className="panel panel-left">
            <div className="panel-section">
              <h2 className="section-title"><span className="section-icon">📄</span> Upload PDF</h2>

              <div
                id="pdf-dropzone"
                className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${file ? 'dropzone-has-file' : ''}`}
                onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
              >
                <input ref={fileInputRef} type="file" accept=".pdf" onChange={(e) => handleFileSelect(e.target.files[0])} style={{ display: 'none' }} />
                {file ? (
                  <div className="dropzone-file-info">
                    <span className="file-icon">📎</span>
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
                  </div>
                ) : (
                  <div className="dropzone-prompt">
                    <span className="upload-icon-emoji">⬆️</span>
                    <p>Drag & drop your PDF here</p>
                    <p className="dropzone-hint">or click to browse</p>
                  </div>
                )}
              </div>

              <button id="upload-btn" className="btn btn-primary btn-full" onClick={handleUpload} disabled={uploading || !file}>
                {uploading ? (<><span className="spinner" /> Indexing with BGE-large embeddings…</>) : 'Upload & Process'}
              </button>

              {uploadError && <div className="error-msg">{uploadError}</div>}
            </div>

            {documents.length > 0 && (
              <div className="panel-section">
                <h2 className="section-title"><span className="section-icon">📚</span> Document Library</h2>
                <div className="doc-library">
                  {documents.map((doc) => (
                    <div key={doc.doc_id} id={`doc-card-${doc.doc_id}`}
                      className={`doc-card ${doc.doc_id === activeDocId ? 'doc-card-active' : ''}`}
                      onClick={() => { setActiveDocId(doc.doc_id); setAnswer(null); setAskError(''); }}>
                      <div className="doc-card-icon">📄</div>
                      <div className="doc-card-info">
                        <span className="doc-card-name">{doc.filename}</span>
                        <span className="doc-card-meta">{doc.num_chunks} chunks · {doc.num_pages} pages · {doc.embedding_time_ms}ms</span>
                      </div>
                      {doc.doc_id === activeDocId && <span className="doc-active-badge">Active</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="panel panel-right">
            {documents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <h3>Upload a PDF to get started</h3>
                <p>Your AI-powered Q&A assistant is ready. Upload a document on the left to begin asking questions.</p>
              </div>
            ) : (
              <>
                <div className="panel-section">
                  <h2 className="section-title">
                    <span className="section-icon">💬</span> Ask a Question
                    {activeDoc && <span className="active-doc-badge">Querying: {activeDoc.filename}</span>}
                  </h2>

                  <div className="question-input-group">
                    <textarea id="question-input" className="question-input"
                      placeholder="What would you like to know about this document?"
                      value={question} onChange={(e) => { setQuestion(e.target.value); if (answer) setAnswer(null); }}
                      onKeyDown={handleKeyDown} rows={3} />
                    <button id="ask-btn" className="btn btn-accent" onClick={handleAsk} disabled={asking || !question.trim()}>
                      {asking ? (<><span className="spinner" /> Searching…</>) : (<>Ask <span className="btn-arrow">→</span></>)}
                    </button>
                  </div>

                  {asking && <div className="search-status"><span className="spinner spinner-sm" /> Searching with BGE + Reranker…</div>}
                  {askError && <div className="error-msg">{askError}</div>}
                </div>

                {answer && (
                  <div className="panel-section answer-section animate-in">
                    <h2 className="section-title"><span className="section-icon">✨</span> Answer</h2>

                    <div className="answer-box">
                      {answer.answer.split('\n').filter(line => line.trim()).map((line, i) => (
                        <p key={i} className={`answer-text ${line.trim().startsWith('•') ? 'answer-bullet' : ''}`}>
                          {line}
                        </p>
                      ))}
                    </div>

                    <div className="answer-meta-row">
                      <div className="confidence-row">
                        <span className="confidence-label">Confidence</span>
                        <div className="confidence-bar-track">
                          <div className="confidence-bar-fill" style={{ width: `${confidencePercent}%` }} />
                        </div>
                        <span className="confidence-value">{confidencePercent}%</span>
                      </div>

                      {answer.total_time_ms && (
                        <div className="time-badge">⚡ {answer.total_time_ms}ms total · {answer.retrieval_time_ms}ms retrieval</div>
                      )}
                    </div>

                    {answer.method && (
                      <div className="method-badge">
                        <span className="badge-dot" />
                        Strategy: {answer.method === 'semantic_retrieval' ? 'Semantic Retrieval (comprehensive)' :
                                   answer.method === 'extractive_qa' ? 'Extractive QA (precise)' :
                                   answer.method}
                      </div>
                    )}

                    <div className="pipeline-badge">
                      <span className="badge-dot" />
                      {answer.retrieval_pipeline || 'BGE-large → Reranker → DeBERTa-v3-large'}
                    </div>

                    {answer.source_chunks && answer.source_chunks.length > 0 && (
                      <div className="source-section">
                        <button id="toggle-sources-btn" className="sources-toggle" onClick={() => setShowSources(!showSources)}>
                          {showSources ? '▾' : '▸'} Source Chunks ({answer.source_chunks.length})
                        </button>
                        {showSources && (
                          <div className="sources-list animate-in">
                            {answer.source_chunks.map((chunk, i) => (
                              <div key={i} className="source-chunk">
                                <span className="chunk-label">Chunk {i + 1}</span>
                                <p className="chunk-text">{chunk}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <footer className="app-footer">
          <p>
            Powered by <span className="footer-model">BAAI/bge-large-en-v1.5</span> ·{' '}
            <span className="footer-model">BGE-reranker-base</span> ·{' '}
            <span className="footer-model">deepset/deberta-v3-large-squad2</span> | LangChain LCEL Pipeline
          </p>
        </footer>
      </div>
    </div>
  );
}


/* ================================================================
   ROOT APP — Routes between Login and Main
   ================================================================ */
function App() {
  const [auth, setAuth] = useState(() => {
    const saved = localStorage.getItem('smartpdf_auth');
    return saved ? JSON.parse(saved) : null;
  });

  const handleLogin = (data) => {
    const authData = {
      token: data.access_token,
      user: { username: data.username, full_name: data.full_name },
    };
    localStorage.setItem('smartpdf_auth', JSON.stringify(authData));
    setAuth(authData);
  };

  const handleLogout = () => {
    localStorage.removeItem('smartpdf_auth');
    setAuth(null);
  };

  if (!auth) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return <MainApp token={auth.token} user={auth.user} onLogout={handleLogout} />;
}

export default App;
