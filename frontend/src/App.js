import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Backend URL; override at build time with REACT_APP_API_BASE
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8100";

/* ================================================================
   LOGIN PAGE COMPONENT
   ================================================================ */
function LoginPage({ onLogin, notice }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState(notice || '');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (isRegister) {
        const res = await axios.post(`${API_BASE}/register`, {
          username, password, full_name: fullName,
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
      setError(err.response?.data?.detail || 'Authentication failed.');
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
              <input id="fullname-input" type="text" placeholder="Enter your full name"
                value={fullName} onChange={(e) => setFullName(e.target.value)} required />
            </div>
          )}
          <div className="form-group">
            <label htmlFor="username-input">Username</label>
            <input id="username-input" type="text" placeholder="Enter username"
              value={username} onChange={(e) => setUsername(e.target.value)} required />
          </div>
          <div className="form-group">
            <label htmlFor="password-input">Password</label>
            <input id="password-input" type="password" placeholder="Enter password"
              value={password} onChange={(e) => setPassword(e.target.value)} required />
          </div>
          {error && <div className="login-error">{error}</div>}
          <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
            {loading ? (<><span className="spinner" /> {isRegister ? 'Creating…' : 'Signing In…'}</>) : (isRegister ? 'Create Account' : 'Sign In')}
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
function MainApp({ token, user, onLogout, onSessionExpired }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [documents, setDocuments] = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [docInfo, setDocInfo] = useState(null);
  const [question, setQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [askError, setAskError] = useState('');
  const [showSources, setShowSources] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Filter state
  const [filterPageStart, setFilterPageStart] = useState('');
  const [filterPageEnd, setFilterPageEnd] = useState('');
  const [filterSections, setFilterSections] = useState([]);
  const [filterTags, setFilterTags] = useState([]);
  const [filterMinQuality, setFilterMinQuality] = useState('');

  const fileInputRef = useRef(null);
  const authHeaders = { Authorization: `Bearer ${token}` };

  // Fetch doc info when active document changes
  useEffect(() => {
    if (activeDocId) {
      axios.get(`${API_BASE}/doc-info/${activeDocId}`, { headers: authHeaders })
        .then(res => setDocInfo(res.data))
        .catch((error) => {
          if (error.response?.status === 401) onSessionExpired();
          setDocInfo(null);
        });
    } else {
      setDocInfo(null);
    }
  }, [activeDocId]);

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
    e.preventDefault(); setDragOver(false);
    handleFileSelect(e.dataTransfer.files[0]);
  }, []);
  const handleDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true); }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleUpload = async () => {
    if (!file) { setUploadError('Please select a PDF file first.'); return; }
    if (file.size > 50 * 1024 * 1024) { setUploadError('File too large. Max 50MB.'); return; }

    const formData = new FormData();
    formData.append('file', file);
    try {
      setUploading(true); setUploadError('');
      const response = await axios.post(`${API_BASE}/upload/`, formData, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 300000,
      });
      const newDoc = {
        doc_id: response.data.doc_id,
        filename: response.data.filename,
        num_chunks: response.data.num_chunks,
        num_pages: response.data.num_pages,
        sections: response.data.sections || [],
        embedding_time_ms: response.data.embedding_time_ms,
      };
      setDocuments((prev) => [...prev, newDoc]);
      setActiveDocId(response.data.doc_id);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (error) {
      if (error.response?.status === 401) {
        onSessionExpired();
      } else if (error.code === 'ECONNABORTED') {
        setUploadError('Upload timed out.');
      } else if (error.response?.status === 503) {
        setUploadError('Models are still loading. Please wait.');
      } else {
        setUploadError(error.response?.data?.detail || 'Upload failed.');
      }
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!activeDocId) { setAskError('Please upload and select a document first.'); return; }
    if (!question.trim()) { setAskError('Please enter a question.'); return; }

    // Build request with filters
    const requestBody = { doc_id: activeDocId, question };
    if (filterPageStart) requestBody.page_start = parseInt(filterPageStart);
    if (filterPageEnd) requestBody.page_end = parseInt(filterPageEnd);
    if (filterSections.length > 0) requestBody.sections = filterSections;
    if (filterTags.length > 0) requestBody.tags = filterTags;
    if (filterMinQuality) requestBody.min_quality = parseFloat(filterMinQuality);

    try {
      setAsking(true); setAskError(''); setAnswer(null);
      const response = await axios.post(`${API_BASE}/ask/`, requestBody, {
        headers: authHeaders, timeout: 120000,
      });
      setAnswer(response.data);
    } catch (error) {
      if (error.response?.status === 401) onSessionExpired();
      else if (error.response?.status === 503) setAskError('Models still loading.');
      else setAskError(error.response?.data?.detail || 'Query failed.');
    } finally {
      setAsking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  };

  const clearFilters = () => {
    setFilterPageStart(''); setFilterPageEnd('');
    setFilterSections([]); setFilterTags([]); setFilterMinQuality('');
  };

  const toggleSection = (section) => {
    setFilterSections(prev =>
      prev.includes(section) ? prev.filter(s => s !== section) : [...prev, section]
    );
  };

  const toggleTag = (tag) => {
    setFilterTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const activeDoc = documents.find((d) => d.doc_id === activeDocId);
  const confidencePercent = answer ? (answer.score * 100).toFixed(1) : 0;
  const hasActiveFilters = filterPageStart || filterPageEnd || filterSections.length > 0 || filterTags.length > 0 || filterMinQuality;

  return (
    <div className="app-wrapper">
      <div className="bg-shape bg-shape-1" />
      <div className="bg-shape bg-shape-2" />
      <div className="app-container">
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
                <p className="header-subtitle">3-Stage Hybrid Filtering Pipeline</p>
              </div>
            </div>
            <div className="user-section">
              <span className="user-name">👋 {user.full_name}</span>
              <button className="btn btn-outline btn-sm" onClick={onLogout}>Logout</button>
            </div>
          </div>
        </header>

        <div className="main-grid">
          {/* Left Panel */}
          <div className="panel panel-left">
            <div className="panel-section">
              <h2 className="section-title"><span className="section-icon">📄</span> Upload PDF</h2>
              <div className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${file ? 'dropzone-has-file' : ''}`}
                onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept=".pdf"
                  onChange={(e) => handleFileSelect(e.target.files[0])} style={{ display: 'none' }} />
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
              <button className="btn btn-primary btn-full" onClick={handleUpload} disabled={uploading || !file}>
                {uploading ? (<><span className="spinner" /> Processing…</>) : 'Upload & Process'}
              </button>
              {uploadError && <div className="error-msg">{uploadError}</div>}
            </div>

            {documents.length > 0 && (
              <div className="panel-section">
                <h2 className="section-title"><span className="section-icon">📚</span> Documents</h2>
                <div className="doc-library">
                  {documents.map((doc) => (
                    <div key={doc.doc_id}
                      className={`doc-card ${doc.doc_id === activeDocId ? 'doc-card-active' : ''}`}
                      onClick={() => { setActiveDocId(doc.doc_id); setAnswer(null); setAskError(''); clearFilters(); }}>
                      <div className="doc-card-icon">📄</div>
                      <div className="doc-card-info">
                        <span className="doc-card-name">{doc.filename}</span>
                        <span className="doc-card-meta">{doc.num_chunks} chunks · {doc.num_pages} pages</span>
                      </div>
                      {doc.doc_id === activeDocId && <span className="doc-active-badge">Active</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Filters Panel */}
            {activeDocId && docInfo && (
              <div className="panel-section">
                <h2 className="section-title">
                  <span className="section-icon">🔍</span> Filters
                  {hasActiveFilters && <span className="filter-active-count">{[filterPageStart || filterPageEnd ? 1 : 0, filterSections.length > 0 ? 1 : 0, filterTags.length > 0 ? 1 : 0, filterMinQuality ? 1 : 0].reduce((a, b) => a + b, 0)} active</span>}
                </h2>
                <button className="filter-toggle-btn" onClick={() => setShowFilters(!showFilters)}>
                  {showFilters ? '▾ Hide Filters' : '▸ Show Filters'}
                </button>

                {showFilters && (
                  <div className="filters-panel animate-in">
                    <div className="filter-group">
                      <label className="filter-label">📑 Page Range (Stage 1 - Pre-filter)</label>
                      <div className="filter-row">
                        <input type="number" className="filter-input" placeholder="From"
                          min="1" max={docInfo.total_pages}
                          value={filterPageStart} onChange={(e) => setFilterPageStart(e.target.value)} />
                        <span className="filter-separator">to</span>
                        <input type="number" className="filter-input" placeholder="To"
                          min="1" max={docInfo.total_pages}
                          value={filterPageEnd} onChange={(e) => setFilterPageEnd(e.target.value)} />
                      </div>
                    </div>

                    {docInfo.sections && docInfo.sections.length > 0 && (
                      <div className="filter-group">
                        <label className="filter-label">📂 Sections (Stage 1 - Pre-filter)</label>
                        <div className="filter-chips">
                          {docInfo.sections.slice(0, 15).map((sec) => (
                            <button key={sec}
                              className={`filter-chip ${filterSections.includes(sec) ? 'filter-chip-active' : ''}`}
                              onClick={() => toggleSection(sec)}>
                              {sec.length > 25 ? sec.substring(0, 25) + '…' : sec}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {docInfo.available_tags && docInfo.available_tags.length > 0 && (
                      <div className="filter-group">
                        <label className="filter-label">🏷️ Tags (Stage 3 - Post-filter)</label>
                        <div className="filter-chips">
                          {docInfo.available_tags.map((tag) => (
                            <button key={tag}
                              className={`filter-chip ${filterTags.includes(tag) ? 'filter-chip-active' : ''}`}
                              onClick={() => toggleTag(tag)}>
                              {tag.replace('_', ' ')}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="filter-group">
                      <label className="filter-label">⭐ Min Quality (Stage 3 - Post-filter)</label>
                      <input type="range" className="filter-range" min="0" max="1" step="0.1"
                        value={filterMinQuality || 0}
                        onChange={(e) => setFilterMinQuality(e.target.value > 0 ? e.target.value : '')} />
                      <span className="filter-range-value">{filterMinQuality || 'Any'}</span>
                    </div>

                    {hasActiveFilters && (
                      <button className="btn btn-outline btn-sm btn-full" onClick={clearFilters}>
                        Clear All Filters
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="panel panel-right">
            {documents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <h3>Upload a PDF to get started</h3>
                <p>Your AI-powered Q&A assistant with 3-stage hybrid filtering is ready.</p>
              </div>
            ) : (
              <>
                <div className="panel-section">
                  <h2 className="section-title">
                    <span className="section-icon">💬</span> Ask a Question
                    {activeDoc && <span className="active-doc-badge">Querying: {activeDoc.filename}</span>}
                  </h2>
                  <div className="question-input-group">
                    <textarea className="question-input"
                      placeholder="What would you like to know about this document?"
                      value={question} onChange={(e) => { setQuestion(e.target.value); if (answer) setAnswer(null); }}
                      onKeyDown={handleKeyDown} rows={3} />
                    <button className="btn btn-accent" onClick={handleAsk} disabled={asking || !question.trim()}>
                      {asking ? (<><span className="spinner" /> Searching…</>) : (<>Ask <span className="btn-arrow">→</span></>)}
                    </button>
                  </div>
                  {hasActiveFilters && (
                    <div className="active-filters-summary">
                      🔍 Filters active: {[
                        filterPageStart || filterPageEnd ? `Pages ${filterPageStart || 1}-${filterPageEnd || '∞'}` : null,
                        filterSections.length > 0 ? `${filterSections.length} section(s)` : null,
                        filterTags.length > 0 ? `${filterTags.length} tag(s)` : null,
                        filterMinQuality ? `Quality ≥ ${filterMinQuality}` : null,
                      ].filter(Boolean).join(' · ')}
                    </div>
                  )}
                  {asking && <div className="search-status"><span className="spinner spinner-sm" /> Running 3-stage pipeline…</div>}
                  {askError && <div className="error-msg">{askError}</div>}
                </div>

                {answer && (
                  <div className="panel-section answer-section animate-in">
                    <h2 className="section-title"><span className="section-icon">✨</span> Answer</h2>
                    <div className="answer-box">
                      {answer.answer_found === false && (
                        <p className="answer-text">{answer.answer}</p>
                      )}
                      {answer.answer_found === false && answer.answer_blocks && answer.answer_blocks.length > 0 && (
                        <div className="related-label">Closest sections (low relevance)</div>
                      )}
                      {answer.answer_blocks && answer.answer_blocks.length > 0 ? (
                        answer.answer_blocks.map((block, i) => (
                          <div key={i} className="answer-group">
                            <div className="answer-group-head">
                              <span className="answer-group-title">{block.heading}</span>
                              <span className="answer-page-badge">p. {block.page}</span>
                            </div>
                            <ul className="answer-list">
                              {block.items.map((item, j) => <li key={j}>{item}</li>)}
                            </ul>
                          </div>
                        ))
                      ) : answer.answer_found === false ? null : (
                        answer.answer.split('\n').filter(line => line.trim()).map((line, i) => (
                          <p key={i} className={`answer-text ${line.trim().startsWith('•') ? 'answer-bullet' : ''}`}>{line}</p>
                        ))
                      )}
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

                    {/* Pipeline Stages Breakdown */}
                    {answer.pipeline_stages && (
                      <div className="pipeline-stages">
                        <div className="stage-item">
                          <span className="stage-label">Stage 1: Pre-filter</span>
                          <span className="stage-value">{answer.pipeline_stages.stage1_prefilter_ms}ms → {answer.pipeline_stages.stage1_chunks_after} chunks</span>
                        </div>
                        <div className="stage-item">
                          <span className="stage-label">Stage 2: Vector + BM25 (RRF)</span>
                          <span className="stage-value">{answer.pipeline_stages.stage2_search_ms}ms → {answer.pipeline_stages.stage2_candidates} candidates</span>
                        </div>
                        <div className="stage-item">
                          <span className="stage-label">Stage 3: Post-filter + Rerank</span>
                          <span className="stage-value">{answer.pipeline_stages.stage3_postfilter_ms}ms + {answer.pipeline_stages.stage3_rerank_ms}ms → {answer.pipeline_stages.stage3_final_docs} results</span>
                        </div>
                      </div>
                    )}

                    {answer.method && (
                      <div className="method-badge">
                        <span className="badge-dot" />
                        {{ semantic_retrieval: 'Semantic Retrieval', not_found: 'Not Found', none: 'No Result' }[answer.method] || 'Extractive QA'}
                      </div>
                    )}

                    <div className="pipeline-badge">
                      <span className="badge-dot" />
                      {answer.retrieval_pipeline || '3-Stage Hybrid Pipeline'}
                    </div>

                    {answer.source_chunks && answer.source_chunks.length > 0 && (
                      <div className="source-section">
                        <button className="sources-toggle" onClick={() => setShowSources(!showSources)}>
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
            3-Stage Hybrid Filtering: <span className="footer-model">Pre-filter</span> →{' '}
            <span className="footer-model">Vector + BM25</span> →{' '}
            <span className="footer-model">Post-filter + Rerank</span> | BGE + DeBERTa
          </p>
        </footer>
      </div>
    </div>
  );
}

/* ================================================================
   ROOT APP
   ================================================================ */
function App() {
  const [auth, setAuth] = useState(() => {
    const saved = localStorage.getItem('smartpdf_auth');
    return saved ? JSON.parse(saved) : null;
  });

  const [notice, setNotice] = useState('');

  const handleLogin = (data) => {
    const authData = {
      token: data.access_token,
      user: { username: data.username, full_name: data.full_name },
    };
    localStorage.setItem('smartpdf_auth', JSON.stringify(authData));
    setNotice('');
    setAuth(authData);
  };

  const handleLogout = () => {
    localStorage.removeItem('smartpdf_auth');
    setAuth(null);
  };

  // Server rejected our token (expired, or signed with a different key): back to login
  const handleSessionExpired = () => {
    setNotice('Your session has expired. Please sign in again.');
    handleLogout();
  };

  if (!auth) return <LoginPage onLogin={handleLogin} notice={notice} />;
  return (
    <MainApp token={auth.token} user={auth.user}
      onLogout={handleLogout} onSessionExpired={handleSessionExpired} />
  );
}

export default App;
