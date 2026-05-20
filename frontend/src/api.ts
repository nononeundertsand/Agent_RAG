export type IngestResponse = {
  corpus_id: string;
  imported_at: string;
  document_count: number;
  title_points: number;
  body_points: number;
  documents: Array<{
    title: string;
    source: string;
    content_kind: string;
    image_count: number;
    body_chunks: number;
  }>;
};

export type ChatResponse = {
  answer: string;
  corpus_id: string;
  quality: {
    doc_count: number;
    max_score: number;
    query_coverage: number;
    source_count: number;
    should_expand: boolean;
    reason: string;
  };
  retrieved_chunks: Array<{
    rank: number;
    score: number;
    type: string;
    title: string;
    source: string;
    document_id: string;
    preview: string;
  }>;
  events: Array<Record<string, unknown>>;
  cache: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    }
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function ingestUrls(payload: { tenant_id: string; urls: string[] }): Promise<IngestResponse> {
  return request<IngestResponse>("/v1/corpora/ingest", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function chat(payload: {
  tenant_id: string;
  corpus_id: string;
  question: string;
  chat_history: Array<{ role: "user" | "assistant"; content: string }>;
}): Promise<ChatResponse> {
  return request<ChatResponse>("/v1/chat", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
