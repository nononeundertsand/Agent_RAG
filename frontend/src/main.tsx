import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, Database, Loader2, MessageSquare, Search, Server, ShieldCheck } from "lucide-react";
import { chat, ChatResponse, ingestUrls, IngestResponse } from "./api";
import "./styles.css";

type ChatItem = { role: "user" | "assistant"; content: string };

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function App() {
  const [tenantId, setTenantId] = useState("default");
  const [urls, setUrls] = useState("");
  const [corpusId, setCorpusId] = useState("");
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<ChatItem[]>([]);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [chatResult, setChatResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState<"ingest" | "chat" | "">("");
  const [error, setError] = useState("");

  const urlList = useMemo(
    () => urls.split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
    [urls]
  );

  async function handleIngest() {
    setError("");
    setLoading("ingest");
    try {
      const result = await ingestUrls({ tenant_id: tenantId, urls: urlList });
      setIngestResult(result);
      setCorpusId(result.corpus_id);
      setChatResult(null);
      setHistory([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading("");
    }
  }

  async function handleChat() {
    if (!question.trim()) return;
    setError("");
    setLoading("chat");
    const nextHistory: ChatItem[] = [...history, { role: "user", content: question }];
    setHistory(nextHistory);
    try {
      const result = await chat({
        tenant_id: tenantId,
        corpus_id: corpusId,
        question,
        chat_history: history
      });
      setChatResult(result);
      setHistory([...nextHistory, { role: "assistant", content: result.answer }]);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setHistory(history);
    } finally {
      setLoading("");
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Search size={28} />
          <div>
            <h1>Agent RAG</h1>
            <p>企业知识库控制台</p>
          </div>
        </div>

        <label>
          租户
          <input value={tenantId} onChange={(event) => setTenantId(event.target.value)} />
        </label>

        <label>
          Corpus ID
          <input value={corpusId} onChange={(event) => setCorpusId(event.target.value)} placeholder="导入后自动填充" />
        </label>

        <div className="navBlock">
          <div><ShieldCheck size={18} /> 租户隔离</div>
          <div><Database size={18} /> Qdrant 检索</div>
          <div><Server size={18} /> FastAPI 服务</div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>RAG 工作台</h2>
            <p>批量导入网页，检索质量评估，查看回答依据与缓存状态。</p>
          </div>
          {loading && <span className="loading"><Loader2 size={18} />处理中</span>}
        </header>

        {error && <div className="error">{error}</div>}

        <section className="panel ingest">
          <div className="panelTitle">
            <BookOpen size={20} />
            <h3>资料导入</h3>
          </div>
          <textarea
            value={urls}
            onChange={(event) => setUrls(event.target.value)}
            placeholder="每行一个 URL，例如：https://example.com/article"
          />
          <button disabled={!urlList.length || loading === "ingest"} onClick={handleIngest}>
            {loading === "ingest" ? "导入中" : `导入 ${urlList.length || 0} 个链接`}
          </button>
        </section>

        {ingestResult && (
          <section className="metricsRow">
            <Metric label="Corpus" value={ingestResult.corpus_id.slice(0, 8)} />
            <Metric label="文档数" value={ingestResult.document_count} />
            <Metric label="正文片段" value={ingestResult.body_points} />
            <Metric label="标题点位" value={ingestResult.title_points} />
          </section>
        )}

        <section className="contentGrid">
          <div className="panel chat">
            <div className="panelTitle">
              <MessageSquare size={20} />
              <h3>问答</h3>
            </div>
            <div className="messages">
              {history.length === 0 && <p className="empty">导入资料后开始提问。</p>}
              {history.map((item, index) => (
                <div key={`${item.role}-${index}`} className={`message ${item.role}`}>
                  {item.content}
                </div>
              ))}
            </div>
            <div className="askBar">
              <input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && handleChat()}
                placeholder="输入你的问题"
                disabled={!corpusId}
              />
              <button disabled={!corpusId || loading === "chat"} onClick={handleChat}>发送</button>
            </div>
          </div>

          <div className="panel debug">
            <div className="panelTitle">
              <Search size={20} />
              <h3>检索质量</h3>
            </div>
            {chatResult ? (
              <>
                <div className="metricsCompact">
                  <Metric label="命中" value={chatResult.quality.doc_count} />
                  <Metric label="最高分" value={chatResult.quality.max_score.toFixed(3)} />
                  <Metric label="覆盖率" value={chatResult.quality.query_coverage.toFixed(3)} />
                  <Metric label="来源" value={chatResult.quality.source_count} />
                </div>
                <div className="cacheLine">缓存：{String(chatResult.cache?.hit ?? false)} / {String(chatResult.cache?.layer ?? "-")}</div>
                <div className="chunks">
                  {chatResult.retrieved_chunks.map((chunk) => (
                    <article key={`${chunk.rank}-${chunk.document_id}`}>
                      <strong>#{chunk.rank} {chunk.title}</strong>
                      <span>{chunk.type} · score {chunk.score}</span>
                      <p>{chunk.preview}</p>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <p className="empty">提交问题后显示召回片段、分数和质量指标。</p>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
