import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Search, ArrowLeft, X, LayoutGrid, Loader2, Database, Download, ChevronLeft, ChevronRight, Scan } from 'lucide-react';
import { searchPages, searchSimilar, getClusters, getClusterDetail, getStatus, startIndexing, getIndexStatus, pageImageUrl, exportSnapshot } from './api';

function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [activeClusterId, setActiveClusterId] = useState(null);
  const [activeCluster, setActiveCluster] = useState(null);
  const [clusterPages, setClusterPages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Results count
  const [topK, setTopK] = useState(10);

  // Lightbox state
  const [lightboxIndex, setLightboxIndex] = useState(null);
  const [lightboxItems, setLightboxItems] = useState([]);

  // Admin + indexing state
  const [adminMode, setAdminMode] = useState(false);
  const [hasGpu, setHasGpu] = useState(false);
  const [canIndex, setCanIndex] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexProgress, setIndexProgress] = useState({
    indexed: 0, total: 0, errors: 0, stage: '',
    scanned_pdfs: 0, scanned_pages: 0, classified_pages: 0,
    visual_pages: 0, scan_complete: false, classify_complete: false,
  });
  const [indexError, setIndexError] = useState(null);
  const pollRef = useRef(null);

  // Connection status
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(false);

  // Load clusters + check GPU on mount
  useEffect(() => {
    getClusters()
      .then(setClusters)
      .catch(() => setClusters([]));
    getStatus()
      .then((s) => {
        setHasGpu(s.gpu);
        setCanIndex(s.can_index);
        setAdminMode(s.admin_mode || false);
        setBackendReady(true);
        setBackendError(false);
      })
      .catch(() => {
        setBackendError(true);
      });
    // Check if indexing is already running
    getIndexStatus()
      .then((s) => {
        if (s.running) {
          setIndexing(true);
          setIndexProgress({
            indexed: s.indexed || 0, total: s.total || 0, errors: s.errors || 0, stage: s.stage || '',
            scanned_pdfs: s.scanned_pdfs || 0, scanned_pages: s.scanned_pages || 0,
            classified_pages: s.classified_pages || 0, visual_pages: s.visual_pages || 0,
            scan_complete: s.scan_complete || false, classify_complete: s.classify_complete || false,
          });
        }
      })
      .catch(() => {});
  }, []);

  // Poll index status while indexing
  useEffect(() => {
    if (!indexing) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await getIndexStatus();
        setIndexProgress({
            indexed: s.indexed || 0, total: s.total || 0, errors: s.errors || 0, stage: s.stage || '',
            scanned_pdfs: s.scanned_pdfs || 0, scanned_pages: s.scanned_pages || 0,
            classified_pages: s.classified_pages || 0, visual_pages: s.visual_pages || 0,
            scan_complete: s.scan_complete || false, classify_complete: s.classify_complete || false,
          });
        if (!s.running) {
          setIndexing(false);
          clearInterval(pollRef.current);
          if (s.error_message) {
            setIndexError('Indexing failed. Check server logs.');
          } else {
            setIndexError(null);
          }
          // Refresh clusters after indexing
          getClusters().then(setClusters).catch(() => {});
        }
      } catch {
        // Ignore transient fetch errors during polling
      }
    }, 2000);
    return () => clearInterval(pollRef.current);
  }, [indexing]);

  const handleStartIndex = async () => {
    setIndexError(null);
    try {
      await startIndexing();
      setIndexing(true);
      setIndexProgress({
        indexed: 0, total: 0, errors: 0, stage: 'pipeline',
        scanned_pdfs: 0, scanned_pages: 0, classified_pages: 0,
        visual_pages: 0, scan_complete: false, classify_complete: false,
      });
    } catch (e) {
      setIndexError(e.message);
    }
  };

  // Search on Enter key
  const handleSearch = useCallback(async (overrideTopK) => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setLightboxIndex(null);
    try {
      const results = await searchPages(q, overrideTopK ?? topK);
      setSearchResults(results);
    } catch (e) {
      setError(e.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, topK]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  const clearSearch = () => {
    setQuery('');
    setSearchResults([]);
    setError(null);
    setLightboxIndex(null);
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const url = await exportSnapshot();
      window.open(url, '_blank');
    } catch (e) {
      setError(e.message);
    } finally {
      setExporting(false);
    }
  };

  // Load cluster detail when a cluster is selected
  const openCluster = async (clusterId) => {
    setActiveClusterId(clusterId);
    setLoading(true);
    setError(null);
    setLightboxIndex(null);
    try {
      const detail = await getClusterDetail(clusterId);
      setActiveCluster(detail);
      setClusterPages(detail.pages || []);
    } catch (e) {
      setError(e.message);
      setClusterPages([]);
    } finally {
      setLoading(false);
    }
  };

  const closeCluster = () => {
    setActiveClusterId(null);
    setActiveCluster(null);
    setClusterPages([]);
    setLightboxIndex(null);
  };

  const handleFindSimilar = useCallback(async (pointId, label) => {
    setLightboxIndex(null);
    setLoading(true);
    setError(null);
    setActiveClusterId(null);
    setActiveCluster(null);
    setClusterPages([]);
    setQuery(`Similar to ${label}`);
    try {
      const results = await searchSimilar(pointId, topK);
      setSearchResults(results);
    } catch (e) {
      setError(e.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  }, [topK]);

  const hasSearchResults = searchResults.length > 0 || query.trim();
  const COLORS = [
    'text-blue-400', 'text-purple-400', 'text-emerald-400',
    'text-orange-400', 'text-rose-400', 'text-indigo-400',
    'text-cyan-400', 'text-amber-400', 'text-teal-400', 'text-pink-400',
  ];

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-gray-300 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">

      {/* Background Gradient Spotlights */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-900/20 rounded-full blur-3xl opacity-50"></div>
          <div className="absolute top-20 right-0 w-[500px] h-[500px] bg-blue-900/10 rounded-full blur-3xl opacity-30"></div>
      </div>

      <div className="relative flex flex-col items-center py-16 px-4 sm:px-6 lg:px-8">

        {/* Header & Search */}
        <div className="w-full max-w-4xl space-y-10 mb-16 text-center">
          <h1 className="text-4xl sm:text-5xl font-extralight tracking-tight text-white">
            PD<span className="font-semibold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">files</span>
          </h1>

          <div className="relative max-w-2xl mx-auto group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl opacity-30 group-hover:opacity-50 blur transition duration-500"></div>
            <div className="relative flex items-center bg-[#121214] rounded-2xl border border-white/10 shadow-2xl">
              <div className="pl-5 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-500 group-hover:text-indigo-400 transition-colors" />
              </div>
              <input
                type="text"
                className="block w-full pl-4 pr-12 py-4 bg-transparent border-none text-white placeholder-gray-600 focus:outline-none focus:ring-0 text-lg"
                placeholder="Search documents..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              {query && (
                <button
                  onClick={clearSearch}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center cursor-pointer text-gray-500 hover:text-gray-300"
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>
          </div>

          {/* Admin Controls */}
          {adminMode && (
            <div className="max-w-2xl mx-auto mt-4 space-y-2">
              <div className="flex items-center justify-center gap-3">
                {canIndex && (
                  <button
                    onClick={handleStartIndex}
                    disabled={indexing}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20 hover:border-indigo-500/40"
                  >
                    {indexing ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Database className="h-4 w-4" />
                    )}
                    {indexing ? 'Indexing...' : 'Index Documents'}
                  </button>
                )}
                <button
                  onClick={handleExport}
                  disabled={exporting}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 hover:border-emerald-500/40"
                >
                  {exporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  {exporting ? 'Creating...' : 'Export Snapshot'}
                </button>
              </div>
              {indexing && (
                <div className="space-y-2">
                  {/* Primary metric: pages searchable */}
                  <p className="text-sm font-medium text-white">
                    {indexProgress.indexed.toLocaleString()} pages searchable
                  </p>

                  {/* Progress bar based on indexed / visual_pages */}
                  {indexProgress.total > 0 && (
                    <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500 transition-all duration-500"
                        style={{ width: `${Math.min(100, (indexProgress.indexed / indexProgress.total) * 100)}%` }}
                      />
                    </div>
                  )}

                  {/* Three-stage status */}
                  <div className="flex gap-4 text-xs font-mono text-gray-500">
                    <span className={indexProgress.scan_complete ? 'text-emerald-500' : 'text-indigo-400'}>
                      {indexProgress.scan_complete ? '\u2713' : '\u25CB'} Scan {indexProgress.scanned_pages.toLocaleString()} pages
                    </span>
                    <span className={indexProgress.classify_complete ? 'text-emerald-500' : indexProgress.classified_pages > 0 ? 'text-indigo-400' : 'text-gray-600'}>
                      {indexProgress.classify_complete ? '\u2713' : '\u25CB'} Classify {indexProgress.classified_pages.toLocaleString()} pages
                    </span>
                    <span className={indexProgress.indexed > 0 ? 'text-indigo-400' : 'text-gray-600'}>
                      {'\u25CB'} Index {indexProgress.indexed.toLocaleString()}/{indexProgress.total.toLocaleString()} pages
                    </span>
                  </div>

                  {indexProgress.errors > 0 && (
                    <p className="text-xs text-red-400 font-mono">{indexProgress.errors} errors</p>
                  )}
                </div>
              )}
              {indexError && (
                <p className="text-xs text-red-400">{indexError}</p>
              )}
            </div>
          )}
        </div>

        {/* Connection status */}
        {backendError && (
          <div className="w-full max-w-2xl mb-4 px-4 py-2 rounded-lg bg-red-500/5 border border-red-500/10 text-center">
            <p className="text-xs text-red-300/70">Cannot connect to backend</p>
          </div>
        )}
        {!backendReady && !backendError && (
          <div className="w-full max-w-2xl mb-4 px-4 py-2 rounded-lg bg-amber-500/5 border border-amber-500/10 text-center">
            <p className="text-xs text-amber-300/70">Loading model... this takes 2-3 minutes on first start</p>
          </div>
        )}

        {/* Indexing banner */}
        {indexing && indexProgress.indexed > 0 && (
          <div className="w-full max-w-6xl mb-4 px-4 py-2 rounded-lg bg-indigo-500/5 border border-indigo-500/10 text-center">
            <p className="text-xs text-indigo-300/70">Indexing in progress — results may be partial</p>
          </div>
        )}

        {/* Main Content */}
        <div className="w-full max-w-6xl z-10">

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-8 w-8 text-indigo-400 animate-spin" />
            </div>
          )}

          {/* Error State */}
          {error && !loading && (
            <div className="text-center py-10 text-red-400">
              <p>{error}</p>
            </div>
          )}

          {/* Search Results */}
          {!loading && hasSearchResults && !activeClusterId && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between px-2">
                <h2 className="text-lg font-medium text-gray-400">
                  Results for "<span className="text-white">{query}</span>"
                </h2>
                <div className="flex items-center gap-3">
                  <select
                    value={topK}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setTopK(val);
                      handleSearch(val);
                    }}
                    className="text-xs font-mono text-gray-400 bg-gray-900 px-2 py-1 rounded-md border border-gray-800 focus:outline-none focus:border-indigo-500/50 cursor-pointer appearance-none"
                  >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </div>
              </div>

              {searchResults.length > 0 ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                  {searchResults.map((r, i) => (
                    <PageCard key={r.point_id} result={r} onClick={() => { setLightboxItems(searchResults); setLightboxIndex(i); }} />
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-20 text-gray-600 space-y-4">
                  <Search className="h-12 w-12 opacity-20" />
                  <p>No documents found</p>
                </div>
              )}
            </div>
          )}

          {/* Cluster Detail View */}
          {!loading && activeClusterId && activeCluster && (
            <div className="space-y-8 animate-in fade-in zoom-in-95 duration-300">
              <div className="flex items-center justify-between border-b border-gray-800 pb-6">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={closeCluster}
                    className="p-2 -ml-2 rounded-full hover:bg-white/5 text-gray-400 hover:text-white transition-all"
                  >
                    <ArrowLeft className="h-6 w-6" />
                  </button>
                  <div>
                    <h2 className="text-3xl font-light text-white">{activeCluster.label || `Cluster ${activeCluster.cluster_id}`}</h2>
                    <p className="text-sm text-gray-500 mt-1">Exploration Mode</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2 bg-gray-900/50 px-3 py-1.5 rounded-lg border border-gray-800/50">
                  <LayoutGrid className="h-4 w-4 text-indigo-400" />
                  <span className="text-sm font-medium text-gray-300">
                    {activeCluster.page_count} pages
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                {clusterPages.slice(0, 50).map((p, i) => (
                  <PageCard key={p.page_id} result={{ page_id: p.page_id, point_id: p.page_id, score: null }} onClick={() => { setLightboxItems(clusterPages.slice(0, 50).map(cp => ({ page_id: cp.page_id, point_id: cp.page_id, score: null }))); setLightboxIndex(i); }} />
                ))}
              </div>
            </div>
          )}

          {/* Home: Cluster Grid */}
          {!loading && !hasSearchResults && !activeClusterId && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 animate-in fade-in duration-700">
              {clusters.map((cluster, idx) => (
                <button
                  key={cluster.cluster_id}
                  onClick={() => openCluster(cluster.cluster_id)}
                  className="group relative h-64 flex flex-col items-center justify-end p-6 rounded-3xl text-center transition-all duration-300 hover:-translate-y-2"
                >
                  {/* Thumbnail Stack */}
                  <div className="absolute inset-0 top-6 flex justify-center pointer-events-none">
                    {(cluster.representative_ids || []).slice(0, 3).map((eid, i) => {
                      let rotate = 0, translateX = 0, zIndex = 10, scale = 1;
                      if (i === 0) { rotate = -2; zIndex = 30; }
                      else if (i === 1) { rotate = 8; translateX = 15; zIndex = 20; scale = 0.95; }
                      else { rotate = -12; translateX = -15; zIndex = 10; scale = 0.9; }

                      return (
                        <div
                          key={eid}
                          className="absolute w-20 h-28 bg-[#1F1F22] border border-gray-700 rounded-md shadow-xl overflow-hidden group-hover:-translate-y-2 transition-transform duration-500 ease-out"
                          style={{ transform: `translateX(${translateX}px) rotate(${rotate}deg) scale(${scale})`, zIndex, top: i * 4 + 'px' }}
                        >
                          <img
                            src={pageImageUrl(eid)}
                            alt={eid}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        </div>
                      );
                    })}
                  </div>

                  <div className="relative z-40 mt-auto w-full">
                     <h3 className={`text-xl font-medium mb-1 group-hover:text-white transition-colors ${COLORS[idx % COLORS.length]}`}>
                      {cluster.label || `Cluster ${cluster.cluster_id}`}
                    </h3>
                    <p className="text-xs font-mono text-gray-600 group-hover:text-gray-500 transition-colors">
                      {cluster.page_count} pages
                    </p>
                  </div>
                </button>
              ))}
            </div>
          )}

        </div>
      </div>

      {/* Lightbox */}
      {lightboxIndex !== null && lightboxItems[lightboxIndex] && (
        <Lightbox
          items={lightboxItems}
          index={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
          onNavigate={setLightboxIndex}
          onFindSimilar={handleFindSimilar}
        />
      )}
    </div>
  );
}

function Lightbox({ items, index, onClose, onNavigate, onFindSimilar }) {
  const item = items[index];
  const hasPrev = index > 0;
  const hasNext = index < items.length - 1;
  const sourcePath = item.source_path || item.pdf_path;

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft' && hasPrev) onNavigate(index - 1);
      else if (e.key === 'ArrowRight' && hasNext) onNavigate(index + 1);
    };
    window.addEventListener('keydown', handleKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handleKey);
      document.body.style.overflow = '';
    };
  }, [index, hasPrev, hasNext, onClose, onNavigate]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm" onClick={onClose}>
      {/* Close button */}
      <button onClick={onClose} className="absolute top-4 right-4 z-50 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors">
        <X className="h-6 w-6" />
      </button>

      {/* Prev arrow */}
      {hasPrev && (
        <button
          onClick={(e) => { e.stopPropagation(); onNavigate(index - 1); }}
          className="absolute left-4 z-50 p-3 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
        >
          <ChevronLeft className="h-6 w-6" />
        </button>
      )}

      {/* Next arrow */}
      {hasNext && (
        <button
          onClick={(e) => { e.stopPropagation(); onNavigate(index + 1); }}
          className="absolute right-4 z-50 p-3 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
        >
          <ChevronRight className="h-6 w-6" />
        </button>
      )}

      {/* Content */}
      <div className="flex flex-col items-center max-h-[90vh] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
        <img
          src={pageImageUrl(item.point_id)}
          alt={item.page_id}
          className="max-h-[80vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
        />
        {/* Metadata bar */}
        <div className="flex items-center gap-4 mt-4 px-4 py-2 bg-white/5 rounded-lg border border-white/10 text-sm">
          <span className="text-white font-medium">{item.page_id}</span>
          {item.page_index != null && (
            <span className="text-gray-400">Page {item.page_index + 1}{item.total_pages ? ` of ${item.total_pages}` : ''}</span>
          )}
          {item.score != null && (
            <span className="text-indigo-400">Score: {item.score.toFixed(4)}</span>
          )}
          <span className="text-gray-500">{index + 1} / {items.length}</span>
          {sourcePath && (
            <span className="max-w-[28rem] truncate text-gray-500 font-mono text-xs" title={sourcePath}>
              {sourcePath}
            </span>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); onFindSimilar(item.point_id, item.page_id); }}
            className="ml-2 p-1.5 rounded bg-white/10 hover:bg-indigo-500/30 text-white hover:text-indigo-300 transition-colors"
            title="Find Similar"
          >
            <Scan className="h-4 w-4" />
          </button>
          <a
            href={pageImageUrl(item.point_id)}
            download={`${item.page_id}.jpg`}
            className="p-1.5 rounded bg-white/10 hover:bg-white/20 text-white transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            <Download className="h-4 w-4" />
          </a>
        </div>
      </div>
    </div>
  );
}

const PageCard = ({ result, onClick }) => (
  <div onClick={onClick} className="group relative flex flex-col bg-[#161618] rounded-xl border border-gray-800 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300 overflow-hidden cursor-pointer hover:-translate-y-1">
    <div className="aspect-[4/3] bg-[#1C1C1E] relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <img
        src={pageImageUrl(result.point_id)}
        alt={result.page_id}
        className="w-full h-full object-cover relative z-10"
        loading="lazy"
      />
    </div>
    <div className="p-4 bg-[#161618] border-t border-gray-800 group-hover:border-gray-700 transition-colors">
      <h3 className="text-sm font-medium text-gray-300 group-hover:text-white truncate transition-colors" title={result.page_id}>
        {result.page_id}
      </h3>
      {result.score != null && (
        <p className="text-[10px] text-gray-600 mt-2 uppercase tracking-wider font-semibold group-hover:text-indigo-400/70 transition-colors">
          Score: {result.score.toFixed(4)}
        </p>
      )}
    </div>
  </div>
);

export default App;
