const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000";

async function safeJson(res) {
  try { return await res.json(); } catch { return null; }
}

export async function uploadExcel(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/upload_excel`, { method: "POST", body: form });
  if (!res.ok) {
    const msg = await safeJson(res);
    throw new Error(msg?.detail || "Upload failed.");
  }
  return res.json();
}

export async function trainModel() {
  const res = await fetch(`${API_BASE_URL}/process_train`, { method: "POST" });
  if (!res.ok) {
    const msg = await safeJson(res);
    throw new Error(msg?.detail || "Training failed.");
  }
  return res.json();
}

export async function predictHotspots(topK = 5) {
  const res = await fetch(`${API_BASE_URL}/predict_hotspots?top_k=${encodeURIComponent(topK)}`);
  if (!res.ok) {
    const msg = await safeJson(res);
    throw new Error(msg?.detail || "Prediction failed.");
  }
  return res.json();
}
