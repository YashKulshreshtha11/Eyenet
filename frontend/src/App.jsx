import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, Eye, Activity, FileText,
  AlertCircle, CheckCircle2, ChevronRight,
  Download, Image as ImageIcon, BrainCircuit, X,
  Scan, Zap, Database, Clock, Target, ArrowLeft, MapPin,
  RefreshCw, AlertTriangle, BarChart2, Layers, Server, FolderOpen, Gauge, Trash2, ShieldAlert
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from 'recharts';

const API_BASE = (import.meta.env.VITE_API_BASE || "/api/v1").replace(/\/+$/, '');

/* ── Tiny helpers ─────────────────────────────────────────── */
const cls = (...args) => args.filter(Boolean).join(' ');

const INDIAN_STATES = [
  'Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Gujarat'
];

const SEVERITY_MAP = {
  normal: { color: 'var(--green)', bg: 'var(--green-10)', label: 'NORMAL' },
  diabeticretinopathy: { color: 'var(--red)', bg: 'var(--red-10)', label: 'DIABETIC RETINOPATHY' },
  glaucoma: { color: 'var(--amber)', bg: 'var(--amber-10)', label: 'GLAUCOMA' },
  cataract: { color: 'var(--blue)', bg: 'var(--blue-10)', label: 'CATARACT' },
};

const toSeverityKey = (value = '') => String(value).toLowerCase().replace(/[\s_]+/g, '');
const getSeverity = (value = '') =>
  SEVERITY_MAP[toSeverityKey(value)] || {
    color: 'var(--cyan)',
    bg: 'var(--cyan-10)',
    label: String(value || 'UNKNOWN').toUpperCase().replace(/_/g, ' '),
  };

/* ── Animated scan line ──────────────────────────────────── */
const ScanLine = () => (
  <div className="scan-line-wrap">
    <div className="scan-line" />
  </div>
);

/* ── Corner brackets ─────────────────────────────────────── */
const Brackets = ({ color = 'var(--cyan)', size = 16, thickness = 2 }) => (
  <>
    {[['tl', '0,0'], ['tr', '0,100'], ['bl', '100,0'], ['br', '100,100']].map(([pos, origin]) => (
      <div key={pos} className={`bracket bracket-${pos}`}
        style={{ '--bcolor': color, '--bsize': `${size}px`, '--bthick': `${thickness}px` }} />
    ))}
  </>
);

/* ── Stat pill ───────────────────────────────────────────── */
const StatPill = ({ icon: Icon, label, value, accent }) => (
  <div className="stat-pill" style={{ '--accent': accent }}>
    <div className="stat-pill-icon"><Icon size={14} /></div>
    <div>
      <div className="stat-pill-val">{value}</div>
      <div className="stat-pill-label">{label}</div>
    </div>
  </div>
);

/* ── Confidence ring ─────────────────────────────────────── */
const ConfidenceRing = ({ value, color }) => {
  const radius = 54;
  const circ = 2 * Math.PI * radius;
  const dash = (value / 100) * circ;
  return (
    <svg width="140" height="140" className="conf-ring">
      <defs>
        <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="1" />
          <stop offset="100%" stopColor="var(--cyan)" stopOpacity="0.6" />
        </linearGradient>
      </defs>
      <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="10" />
      <circle cx="70" cy="70" r={radius} fill="none" stroke="url(#ringGrad)" strokeWidth="10"
        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
        transform="rotate(-90 70 70)" style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1)' }} />
      <text x="70" y="65" textAnchor="middle" fill="white" fontSize="20" fontWeight="700"
        fontFamily="'Syne', sans-serif">{value.toFixed(1)}%</text>
      <text x="70" y="83" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9"
        fontFamily="'DM Mono', monospace" letterSpacing="2">CONFIDENCE</text>
    </svg>
  );
};

/* ── Custom tooltip for bar chart ───────────────────────── */
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{payload[0].payload.name}</div>
      <div className="chart-tooltip-val">{payload[0].value.toFixed(2)}%</div>
    </div>
  );
};

/* ── Main App ────────────────────────────────────────────── */
const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyPage, setHistoryPage] = useState(0);
  const [hasMoreHistory, setHasMoreHistory] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analyze');
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [confirmingDelete, setConfirmingDelete] = useState(null); // stores record object
  const [confirmingDeleteAll, setConfirmingDeleteAll] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState(0);
  const [health, setHealth] = useState({ ready: false, modelLoaded: false, dbAlive: false });
  const [patientState, setPatientState] = useState('Delhi');
  const fileInputRef = useRef(null);
  const progressRef = useRef(null);
  const analyzeTimeoutRef = useRef(null);

  const fetchHistory = useCallback(async (reset = false) => {
    if (loadingHistory) return;
    setLoadingHistory(true);
    const page = reset ? 0 : historyPage;
    const limit = 30;
    const skip = page * limit;

    try {
      const resp = await fetch(`${API_BASE}/history?limit=${limit}&skip=${skip}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const newRecords = data.records || [];

      if (reset) {
        setHistory(newRecords);
        setHistoryPage(1);
      } else {
        setHistory(prev => [...prev, ...newRecords]);
        setHistoryPage(page + 1);
      }

      setHasMoreHistory(newRecords.length === limit);
    } catch (err) {
      console.error("Failed to fetch history", err);
    } finally {
      setLoadingHistory(false);
    }
  }, [historyPage, loadingHistory]);

  const deleteRecordRequest = (record) => {
    setConfirmingDelete(record);
  };

  const executeDelete = async (recordId) => {
    try {
      const resp = await fetch(`${API_BASE}/record/${recordId}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error("Failed to delete record");
      setHistory(prev => prev.filter(r => r.id !== recordId));
      if (selectedRecord?.id === recordId) setSelectedRecord(null);
      setConfirmingDelete(null);
    } catch (err) {
      console.error(err);
      alert("Error deleting record. Please try again.");
    }
  };

  const fetchFullRecord = async (item) => {
    // Optimistically show with whatever data we have, then enrich with heatmap
    setSelectedRecord(item);
    try {
      const resp = await fetch(`${API_BASE}/record/${item.id || item._id}`);
      if (!resp.ok) return;
      const full = await resp.json();
      // Map _id -> id for consistency
      if (full._id && !full.id) full.id = full._id;
      setSelectedRecord(full);
    } catch (err) {
      console.error("Failed to fetch full record", err);
    }
  };

  const executeDeleteAll = async () => {
    try {
      const resp = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
      if (!resp.ok) throw new Error("Failed to clear history");
      setHistory([]);
      setSelectedRecord(null);
      setConfirmingDeleteAll(false);
    } catch (err) {
      console.error(err);
      alert("Error clearing history. Please try again.");
    }
  };

  const fetchHealth = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/health`);
      if (!resp.ok) return;
      const data = await resp.json();
      setHealth({
        ready: Boolean(data.model_loaded && data.weights_loaded),
        modelLoaded: Boolean(data.model_loaded),
        dbAlive: Boolean(data.db_alive),
      });
    } catch (err) {
      console.error("Failed to fetch health", err);
    }
  }, []);

  useEffect(() => {
    fetchHistory(true);
    fetchHealth();
    const healthInterval = setInterval(fetchHealth, 30000);
    return () => {
      clearInterval(healthInterval);
    };
  }, [fetchHealth]);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0] || e.dataTransfer?.files?.[0];
    if (!selected) return;
    if (!selected.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG, BMP, TIFF, WEBP).');
      return;
    }
    setFile(selected);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(selected);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFileChange(e);
  };

  const runAnalysis = async () => {
    if (!file) return;
    if (!health.modelLoaded) {
      setError('Model is not ready yet. Please wait for backend initialization and try again.');
      return;
    }
    setAnalyzing(true);
    setError(null);
    setAnalyzeProgress(0);

    // Fake progress animation
    let p = 0;
    progressRef.current = setInterval(() => {
      p += Math.random() * 8;
      if (p > 90) { clearInterval(progressRef.current); p = 90; }
      setAnalyzeProgress(Math.min(p, 90));
    }, 200);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('state', patientState);
    try {
      // ✅ NEW — paste this instead
      const resp = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
      if (!resp.ok) {
        let detail = `Server error (${resp.status})`;
        try {
          const errBody = await resp.json();
          detail = errBody.detail || errBody.message || errBody.error || JSON.stringify(errBody);
        } catch (_) {
          detail = await resp.text().catch(() => detail);
        }
        throw new Error(detail);
      }
      const data = await resp.json();
      clearInterval(progressRef.current);
      setAnalyzeProgress(100);
      analyzeTimeoutRef.current = setTimeout(() => {
        setResult(data);
        setAnalyzing(false);
      }, 400);
      fetchHistory();
    } catch (err) {
      clearInterval(progressRef.current);
      setError(err.message);
      setAnalyzing(false);
      setAnalyzeProgress(0);
    }
  };

  const downloadReport = (id) => {
    const recordId = id || result?.record_id;
    if (!recordId) return;
    window.open(`${API_BASE}/report/${recordId}`, '_blank');
  };

  const resetScan = () => {
    clearInterval(progressRef.current);
    clearTimeout(analyzeTimeoutRef.current);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setFile(null); setPreview(null); setResult(null); setError(null);
    setAnalyzing(false);
    setAnalyzeProgress(0);
  };

  const chartColors = ['#00d4e8', '#2979ff', '#f5a623', '#ff3b5c'];
  const diagnosticData = result
    ? Object.entries(result.probabilities).map(([name, value], i) => ({
      name: name.replace(/_/g, ' '), value: +(value * 100).toFixed(2), fill: chartColors[i % chartColors.length]
    }))
    : [];

  const maxConf = result
    ? Math.max(...Object.values(result.probabilities)) * 100
    : 0;

  const severity = result ? getSeverity(result.predicted_class) : null;
  const modelCards = [
    { label: 'EfficientNet-B0', role: 'Backbone Classifier' },
    { label: 'ResNet-50', role: 'Backbone Extractor' },
    { label: 'DenseNet-121', role: 'Backbone Extractor' },
  ];

  return (
    <div className="app-root">
      {/* Background layer */}
      <div className="bg-layer" aria-hidden="true">
        <div className="bg-grid" />
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* Header */}
      <header className="app-header">
        <div className="header-inner container">
          <div className="logo-group">
            <div className="logo-eye">
              <Eye size={20} strokeWidth={1.5} />
              <div className="logo-eye-ring" />
            </div>
            <div>
              <div className="logo-title">EyeNet<span>Elite</span></div>
              <div className="logo-sub">AI Retinal Diagnostics Platform</div>
            </div>
          </div>

          <nav className="main-nav">
            {[
              { id: 'analyze', icon: Scan, label: 'Analysis' },
              { id: 'history', icon: Database, label: 'Records' },
            ].map(({ id, icon: Icon, label }) => (
              <button key={id}
                className={cls('nav-btn', activeTab === id && 'nav-btn--active')}
                onClick={() => { setActiveTab(id); setSelectedRecord(null); }}>
                <Icon size={14} />
                <span>{label}</span>
              </button>
            ))}
          </nav>

          <div className="header-meta">
            <div className={cls('system-status', health.ready ? 'system-status--ok' : 'system-status--warn')}>
              <span className="pulse-dot" />
              <span>{health.ready ? 'Model Ready' : 'Model Initializing'}</span>
            </div>
            <div className="version-tag">v2.0.0</div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="app-main container">
        <AnimatePresence mode="wait">

          {/* ── ANALYZE TAB ── */}
          {activeTab === 'analyze' && !selectedRecord && (
            <motion.div key="analyze"
              initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }} transition={{ duration: 0.4 }}
              className="analyze-layout">

              {/* Left: Upload */}
              <div className="upload-col">
                <div className="panel-label"><Scan size={11} />Input Fundus Image</div>
                <div className="quick-stats">
                  <div className="quick-stat">
                    <Server size={13} />
                    <span>{health.modelLoaded ? 'Model Online' : 'Model Offline'}</span>
                  </div>
                  <div className="quick-stat">
                    <Database size={13} />
                    <span>{health.dbAlive ? 'Database Connected' : 'Database Offline'}</span>
                  </div>
                  <div className="quick-stat">
                    <FolderOpen size={13} />
                    <span>{history.length} Records</span>
                  </div>
                </div>


                <div className={cls('upload-zone', dragOver && 'upload-zone--drag', preview && 'upload-zone--filled')}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => !preview && fileInputRef.current?.click()}>

                  <input ref={fileInputRef} type="file" accept="image/*"
                    onChange={handleFileChange} style={{ display: 'none' }} />

                  <AnimatePresence mode="wait">
                    {!preview ? (
                      <motion.div key="empty" className="upload-empty"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <div className="upload-icon-wrap">
                          <div className="upload-icon-inner">
                            <Upload size={28} strokeWidth={1.2} />
                          </div>
                          <div className="upload-icon-rings">
                            <span /><span /><span />
                          </div>
                        </div>
                        <div className="upload-text">
                          <p className="upload-title">Drop Fundus Scan Here</p>
                          <p className="upload-hint">or click to browse — PNG, JPG, TIFF accepted</p>
                        </div>
                        <div className="upload-formats">
                          {['JPEG', 'PNG', 'TIFF', 'BMP'].map(f => (
                            <span key={f} className="fmt-badge">{f}</span>
                          ))}
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div key="preview" className="preview-wrap"
                        initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}>
                        <div className="preview-img-wrap">
                          <img src={preview} alt="Fundus scan preview" className="preview-img" />
                          <div className="preview-overlay">
                            <Brackets />
                            {!result && !analyzing && <ScanLine />}
                            {analyzing && <div className="processing-pulse" />}
                          </div>
                          {result && (
                            <div className="preview-verdict-badge" style={{ '--vc': severity.color, '--vbg': severity.bg }}>
                              <span className="verdict-dot" />
                              {severity.label}
                            </div>
                          )}
                        </div>
                        <div className="preview-meta">
                          <div className="preview-filename">
                            <ImageIcon size={12} />
                            <span>{file?.name}</span>
                          </div>
                          <button className="reset-btn" onClick={(e) => { e.stopPropagation(); resetScan(); }}>
                            <X size={12} /> Reset
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Analyze button */}
                {preview && !result && (
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                    <button className={cls('analyze-btn', analyzing && 'analyze-btn--loading')}
                      onClick={runAnalysis} disabled={analyzing}>
                      {analyzing ? (
                        <>
                          <RefreshCw size={16} className="spin" />
                          <span>Running Ensemble Models…</span>
                          <div className="analyze-progress" style={{ '--prog': `${analyzeProgress}%` }} />
                        </>
                      ) : (
                        <>
                          <BrainCircuit size={16} />
                          <span>Run Ensemble Diagnostics</span>
                          <Zap size={12} className="analyze-btn-zap" />
                        </>
                      )}
                    </button>
                    {analyzing && (
                      <div className="progress-bar-wrap">
                        <div className="progress-bar" style={{ width: `${analyzeProgress}%` }} />
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Error */}
                <AnimatePresence>
                  {error && (
                    <motion.div className="error-card"
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                      <AlertTriangle size={16} />
                      <p>{error}</p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Model info pills */}
                <div className="model-pills">
                  <div className="panel-label" style={{ marginBottom: '0.75rem' }}><Layers size={11} />Ensemble Architecture</div>
                  {modelCards.map(m => (
                    <div key={m.label} className="model-pill">
                      <div className="model-pill-dot" />
                      <div>
                        <div className="model-pill-name">{m.label}</div>
                        <div className="model-pill-role">{m.role}</div>
                      </div>
                      <CheckCircle2 size={12} className="model-pill-check" />
                    </div>
                  ))}
                </div>
              </div>

              {/* Right: Results */}
              <div className="results-col">
                <AnimatePresence mode="wait">
                  {result ? (
                    <motion.div key="results"
                      initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -16 }} transition={{ duration: 0.5 }}
                      className="results-grid">

                      {/* Verdict card */}
                      <div className="verdict-card" style={{ '--vc': severity.color, '--vbg': severity.bg }}>
                        <div className="verdict-glow" />
                        <div className="verdict-left">
                          <div className="panel-label"><Target size={11} />Diagnostic Verdict</div>
                          <div className="verdict-class">{result.predicted_class.replace(/_/g, ' ')}</div>
                          <div className="verdict-sub">Triple-model ensemble consensus</div>
                          <div className="verdict-stats">
                            <StatPill icon={Activity} label="Models Used" value="3" accent="var(--cyan)" />
                            <StatPill icon={Clock} label="Inference Time" value={`${((result.inference_time_ms || 0) / 1000).toFixed(2)}s`} accent="var(--blue)" />
                            <StatPill icon={Gauge} label="Image Quality" value={result.quality_verdict || 'N/A'} accent={severity.color} />
                          </div>
                          <div className="verdict-actions">
                            <button className="download-btn" onClick={() => downloadReport()}>
                              <Download size={14} />
                              <span>Download PDF Report</span>
                            </button>
                            <button className="rescan-btn" onClick={resetScan}>
                              <RefreshCw size={14} /> New Scan
                            </button>
                          </div>
                        </div>
                        <div className="verdict-right">
                          <ConfidenceRing value={maxConf} color={severity.color} />
                        </div>
                      </div>

                      {/* Heatmap + Chart */}
                      <div className="viz-row">
                        <div className="panel heatmap-panel">
                          <div className="panel-label"><ImageIcon size={11} />Grad-CAM Heatmap</div>
                          <div className="heatmap-wrap">
                            <img
                              src={`data:image/png;base64,${result.overlay_base64}`}
                              alt="Grad-CAM overlay" className="heatmap-img" />
                            <div className="heatmap-overlay">
                              <Brackets color={severity.color} size={12} />
                              <div className="heatmap-label">Region of Interest</div>
                            </div>
                          </div>
                        </div>

                        <div className="panel chart-panel">
                          <div className="panel-label"><BarChart2 size={11} />Probability Distribution</div>
                          <div className="chart-wrap">
                            <ResponsiveContainer width="100%" height={200}>
                              <BarChart data={diagnosticData} barSize={28}
                                margin={{ top: 8, right: 8, left: -28, bottom: 0 }}>
                                <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 9, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                                <YAxis tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 9, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} unit="%" />
                                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                  {diagnosticData.map((entry, i) => (
                                    <Cell key={i} fill={entry.fill} fillOpacity={result.predicted_class.replace(/_/g, ' ') === entry.name ? 1 : 0.4} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>

                          {/* Class probability bars */}
                          <div className="prob-bars">
                            {diagnosticData.map((item, i) => (
                              <div key={i} className="prob-bar-row">
                                <span className="prob-bar-name">{item.name}</span>
                                <div className="prob-bar-track">
                                  <motion.div className="prob-bar-fill"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${item.value}%` }}
                                    transition={{ duration: 0.8, delay: i * 0.1, ease: [0.4, 0, 0.2, 1] }}
                                    style={{ background: item.fill }} />
                                </div>
                                <span className="prob-bar-val">{item.value.toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Clinical note */}
                      <div className="clinical-note">
                        <AlertCircle size={14} />
                        <p>{result.quality_advisory || 'This AI-generated analysis is intended to assist clinical decision-making and should not replace professional ophthalmological assessment.'}</p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div key="empty-results"
                      initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.02 }}
                      className="empty-results cinematic-entrance">
                      <div className="empty-eye-wrap">
                        <div className="empty-eye-glow" />
                        <div className="empty-eye">
                          <Scan size={50} strokeWidth={1} className="glow-text-cyan" />
                          <div className="empty-eye-rings">
                            <span /><span /><span /><span />
                          </div>
                        </div>
                      </div>
                      <h3 className="empty-title">Ready for Diagnostics</h3>
                      <p className="empty-sub">State-of-the-art ensemble analysis for retinal pathologies. <br />Upload a fundus image to begin the deep learning assessment.</p>
                      <div className="empty-features-grid">
                        {[
                          { icon: BrainCircuit, text: 'Triple Ensemble', sub: 'ResNet + EffNet + DenseNet' },
                          { icon: Target, text: 'Grad-CAM XAI', sub: 'Attention Localization' },
                          { icon: FileText, text: 'Expert Reports', sub: 'Detailed PDF Export' },
                          { icon: Activity, text: 'Real-time Processing', sub: 'Millisecond Inference' },
                        ].map(({ icon: Icon, text, sub }) => (
                          <div key={text} className="empty-feature-card">
                            <Icon size={18} className="feature-icon" />
                            <div className="feature-info">
                              <div className="feature-title">{text}</div>
                              <div className="feature-sub">{sub}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {/* ── HISTORY TAB ── */}
          {activeTab === 'history' && !selectedRecord && (
            <motion.div key="history"
              initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }} transition={{ duration: 0.4 }}>

              <div className="history-header">
                <div>
                  <div className="panel-label"><Database size={11} />Diagnostic Records</div>
                  <h2 className="history-title">Patient Analysis History</h2>
                </div>
                <div className="history-actions-group">
                  <button className="delete-all-btn" onClick={() => setConfirmingDeleteAll(true)}>
                    <Trash2 size={13} /> Clear All
                  </button>
                  <button className="refresh-btn" onClick={() => fetchHistory(true)}>
                    <RefreshCw size={13} className={cls(loadingHistory && 'spin')} /> Refresh
                  </button>
                </div>
              </div>

              <div className="history-table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      {['#', 'Image File', 'Diagnosis', 'Confidence', 'Timestamp', 'Actions'].map(h => (
                        <th key={h}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {history.length > 0 ? history.map((item, idx) => {
                      const maxP = Math.max(...Object.values(item.probabilities || {}));
                      const sev = getSeverity(item.predicted_class);
                      return (
                        <tr key={item.id || item._id || idx} onClick={() => fetchFullRecord(item)} className="history-row">
                          <td className="history-idx">{String(idx + 1).padStart(2, '0')}</td>
                          <td className="history-file">
                            <div className="history-file-icon"><ImageIcon size={12} /></div>
                            <span>{item.image_name}</span>
                          </td>
                          <td>
                            <span className="verdict-pill" style={{ '--vc': sev.color, '--vbg': sev.bg }}>
                              <span className="verdict-pip" />
                              {(item.predicted_class || 'Unknown').replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td>
                            <div className="conf-cell">
                              <div className="conf-track">
                                <div className="conf-fill" style={{ width: `${maxP * 100}%`, background: sev.color }} />
                              </div>
                              <span>{(maxP * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="history-time">
                            <Clock size={11} />
                            {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                          </td>
                          <td>
                            <div className="history-actions">
                              <button className="history-action-btn" title="Download Report" onClick={(e) => { e.stopPropagation(); downloadReport(item.id || item._id); }}>
                                <Download size={13} />
                              </button>
                              <button className="history-action-btn history-action-btn--danger" title="Delete Record" onClick={(e) => { e.stopPropagation(); deleteRecordRequest(item); }}>
                                <Trash2 size={13} />
                              </button>
                              <button className="history-action-btn history-action-btn--primary" onClick={(e) => { e.stopPropagation(); fetchFullRecord(item); }}>
                                <ChevronRight size={13} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    }) : (
                      <tr><td colSpan="6" className="history-empty">
                        <Database size={32} strokeWidth={0.8} />
                        <p>No diagnostic records found</p>
                      </td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              {hasMoreHistory && (
                <div className="history-load-more">
                  <button className="load-more-btn" onClick={() => fetchHistory()} disabled={loadingHistory}>
                    {loadingHistory ? <RefreshCw size={14} className="spin" /> : <ChevronRight size={14} className="rotate-90" />}
                    <span>{loadingHistory ? 'Loading...' : 'Load More Records'}</span>
                  </button>
                </div>
              )}
            </motion.div>
          )}

          {/* ── HISTORY DETAIL ── */}
          {selectedRecord && (
            <motion.div key="detail"
              initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>

              <button className="back-btn" onClick={() => setSelectedRecord(null)}>
                <ArrowLeft size={14} /> Back to Records
              </button>

              {(() => {
                const sev = getSeverity(selectedRecord.predicted_class);
                const maxP = Math.max(...Object.values(selectedRecord.probabilities || {}));
                return (
                  <div className="detail-layout">
                    <div className="detail-main panel">
                      <div className="panel-label"><Target size={11} />Diagnostic Results</div>
                      <div className="detail-verdict" style={{ '--vc': sev.color }}>
                        {selectedRecord.predicted_class.replace(/_/g, ' ')}
                      </div>
                      <div className="detail-meta">
                        <span><Clock size={11} /> {new Date(selectedRecord.timestamp).toLocaleString()}</span>
                        <span>ID: <code>{selectedRecord.id}</code></span>
                      </div>
                      <div className="detail-prob-list">
                        {Object.entries(selectedRecord.probabilities || {}).map(([cls, p], i) => {
                          const s = getSeverity(cls);
                          return (
                            <div key={cls} className="detail-prob-row">
                              <div className="detail-prob-header">
                                <span className="detail-prob-name">{cls.replace(/_/g, ' ')}</span>
                                <span className="detail-prob-pct" style={{ color: s.color }}>{(p * 100).toFixed(2)}%</span>
                              </div>
                              <div className="detail-prob-track">
                                <motion.div className="detail-prob-fill"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${p * 100}%` }}
                                  transition={{ duration: 0.9, delay: i * 0.1, ease: [0.4, 0, 0.2, 1] }}
                                  style={{ background: s.color }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <button className="download-btn" style={{ marginTop: '2rem' }}
                        onClick={() => downloadReport(selectedRecord.id)}>
                        <Download size={14} /> Download Full PDF Report
                      </button>
                    </div>

                    <div className="detail-side">
                      <div className="panel">
                        <div className="panel-label"><Activity size={11} />Confidence Score</div>
                        <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem 0' }}>
                          <ConfidenceRing value={maxP * 100} color={sev.color} />
                        </div>
                      </div>
                      <div className="panel archive-panel">
                        <div className="panel-label"><ImageIcon size={11} />Heatmap Archive</div>
                        <div className="archive-viewport">
                          {selectedRecord.overlay_base64 || selectedRecord.gradcam_base64 ? (
                            <div className="archive-img-wrap">
                              <img
                                src={`data:image/png;base64,${selectedRecord.overlay_base64 || selectedRecord.gradcam_base64}`}
                                alt="Diagnostic Heatmap"
                                className="archive-img"
                              />
                              <div className="img-badge">ARCHIVE</div>
                            </div>
                          ) : (
                            <div className="archive-placeholder">
                              <ImageIcon size={28} strokeWidth={0.8} />
                              <p>Heatmap unavailable in archive</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-inner container">
          <div className="footer-left">
            <span className="footer-brand">EyeNet Elite</span>
            <span className="footer-div" />
            <span>AI Retinal Analysis System</span>
          </div>
          <div className="footer-right">
            <div className="footer-label">© 2026 Academic Research Project</div>
          </div>
        </div>
      </footer>

      {/* Deletion Confirmation Modal */}
      <AnimatePresence>
        {confirmingDelete && (
          <div className="modal-root">
            <motion.div
              className="confirm-modal"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            >
              <div className="confirm-panel-header">
                <AlertTriangle size={24} className="text-red" />
                <h3>Confirm Deletion</h3>
              </div>

              <div className="confirm-panel-body">
                <p>You are about to permanently delete the diagnostic record for:</p>
                <div className="confirm-record-info">
                  <ImageIcon size={14} />
                  <span>{confirmingDelete.image_name}</span>
                </div>
                <p className="confirm-warning">This action is irreversible and will remove all associated AI telemetry and analysis reports from the mainframe.</p>
              </div>

              <div className="confirm-panel-actions">
                <button className="confirm-btn confirm-btn--danger" onClick={() => executeDelete(confirmingDelete.id)}>
                  <Trash2 size={16} />
                  <span>Delete Permanently</span>
                </button>
                <button className="confirm-btn confirm-btn--cancel" onClick={() => setConfirmingDelete(null)}>
                  <span>Cancel</span>
                </button>
              </div>

              <button className="confirm-close" onClick={() => setConfirmingDelete(null)}>
                <X size={20} />
              </button>
            </motion.div>

            <motion.div
              className="confirm-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setConfirmingDelete(null)}
            />
          </div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {confirmingDeleteAll && (
          <div className="modal-root">
            <motion.div
              className="confirm-modal confirm-modal--destructive"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            >
              <div className="confirm-panel-header">
                <ShieldAlert size={28} className="text-red" />
                <h3>Wipe History Base?</h3>
              </div>

              <div className="confirm-panel-body">
                <p>You are requesting a <strong>Total Data Clearance</strong> of the diagnostic mainframe.</p>
                <div className="destructive-warning-box">
                  <p>This will permanently erase all {history.length} patient records, including:</p>
                  <ul>
                    <li>Deep Learning Inference Logs</li>
                    <li>Generated Heatmap Telemetry</li>
                  </ul>
                </div>
              </div>

              <div className="confirm-panel-actions">
                <button className="confirm-btn confirm-btn--danger" onClick={executeDeleteAll}>
                  <Zap size={16} />
                  <span>Execute Full Clear</span>
                </button>
                <button className="confirm-btn confirm-btn--cancel" onClick={() => setConfirmingDeleteAll(false)}>
                  <span>Abort Mission</span>
                </button>
              </div>

              <button className="confirm-close" onClick={() => setConfirmingDeleteAll(false)}>
                <X size={20} />
              </button>
            </motion.div>

            <motion.div
              className="confirm-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setConfirmingDeleteAll(false)}
            />
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
