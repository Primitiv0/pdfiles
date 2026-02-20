const API_BASE = import.meta.env.VITE_API_URL || '/api';

export async function searchPages(query, topK = 10) {
  const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&top_k=${topK}`);
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function getClusters() {
  const res = await fetch(`${API_BASE}/clusters`);
  if (!res.ok) throw new Error(`Failed to load clusters: ${res.status}`);
  return res.json();
}

export async function getClusterDetail(clusterId) {
  const res = await fetch(`${API_BASE}/clusters/${clusterId}`);
  if (!res.ok) throw new Error(`Failed to load cluster: ${res.status}`);
  return res.json();
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(`Failed to get status: ${res.status}`);
  return res.json();
}

export async function startIndexing() {
  const res = await fetch(`${API_BASE}/index`, { method: 'POST' });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to start indexing: ${res.status}`);
  }
  return res.json();
}

export async function getIndexStatus() {
  const res = await fetch(`${API_BASE}/index/status`);
  if (!res.ok) throw new Error(`Failed to get index status: ${res.status}`);
  return res.json();
}

export async function searchSimilar(pointId, topK = 10) {
  const res = await fetch(`${API_BASE}/search/similar?point_id=${pointId}&top_k=${topK}`);
  if (!res.ok) throw new Error(`Similar search failed: ${res.status}`);
  return res.json();
}

export async function searchByImage(file, topK = 10) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/search/image?top_k=${topK}`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Image search failed: ${res.status}`);
  return res.json();
}

export function pageImageUrl(pointId) {
  return `${API_BASE}/page/${pointId}/image`;
}

export async function exportSnapshot() {
  const res = await fetch(`${API_BASE}/export`, { method: 'POST' });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to create snapshot: ${res.status}`);
  }
  const { name } = await res.json();
  return `${API_BASE}/export/${name}`;
}
