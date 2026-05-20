# Agent RAG 工程化系统

Agent RAG 已从个人 demo 升级为一个前后端分离的工程化 RAG 应用：后端使用 FastAPI 提供入库、检索、问答和健康检查接口；前端使用 React + Vite 构建知识库工作台；向量数据库使用 Qdrant；模型层支持 DeepSeek/Gemini 分支。

## 当前架构

- Frontend：React + TypeScript + Vite，位于 `frontend/`。
- API Service：FastAPI，入口 `blog_rag/api/app.py`。
- RAG Service：入库、检索评估、召回扩展、答案生成，位于 `blog_rag/services/rag_service.py`。
- Vector DB：Qdrant，存储标题向量和正文 chunk 向量。
- Cache：当前实现进程内 TTL Cache，生产建议替换为 Redis。
- Document Processing：网页/Markdown 解析、图片线索提取、chunk 构建，位于 `blog_rag/document_processing.py`。

## 快速启动

### 后端

```powershell
pip install -r requirements.txt

$env:QDRANT_URL="https://your-qdrant-url"
$env:QDRANT_API_KEY="your-qdrant-key"
$env:MODEL_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="your-deepseek-key"

uvicorn blog_rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

健康检查：

```bash
curl http://localhost:8000/health
```

### 前端

```powershell
cd frontend
npm install
npm run dev
```

前端默认通过 Vite proxy 将 `/api/*` 转发到 `http://localhost:8000`。也可以设置：

```powershell
$env:VITE_API_BASE="http://localhost:8000"
```

## API

### 批量导入网页

```bash
curl -X POST http://localhost:8000/v1/corpora/ingest \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","urls":["https://example.com/a","https://example.com/b"]}'
```

返回核心字段：

- `corpus_id`：后续问答使用的资料集 ID。
- `document_count`：成功入库的文档数。
- `title_points` / `body_points`：写入 Qdrant 的标题点位和正文点位。

### 问答

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","corpus_id":"your-corpus-id","question":"这些资料的核心观点是什么？"}'
```

返回核心字段：

- `answer`：基于检索上下文生成的答案。
- `quality`：检索质量评估。
- `retrieved_chunks`：命中片段、来源、分数。
- `events`：检索、扩展召回、缓存命中等链路事件。
- `cache`：生成缓存状态。

## 前端升级

已移除 Streamlit 作为主前端，改用 React + Vite：

- React：构建企业知识库工作台，适合长期扩展权限、路由、状态管理和组件体系。
- TypeScript：约束 API response 和页面状态，降低前后端联调风险。
- Vite：本地开发启动快，构建产物可直接部署到 Nginx、CDN 或对象存储。
- lucide-react：用于控制台图标，避免手写 SVG。

前端已支持：

- 多 URL 导入。
- 自动接收并填充 `corpus_id`。
- 基于当前 corpus 提问。
- 展示检索质量指标：命中数、最高分、覆盖率、来源数。
- 展示命中 chunk、分数、来源和预览。
- 展示缓存命中状态。

预期效果：

- 相比 Streamlit，生产部署更清晰，前端可独立扩容和缓存静态资源。
- 页面交互延迟主要由 API 决定，前端构建产物可通过 CDN 将静态资源延迟降到毫秒级。
- 更容易接入企业 SSO、审计、权限、埋点和多页面管理。

## 检索评估机制优化

当前实现不只判断“是否返回 chunk”，而是使用质量指标决定是否进入失败恢复流程。

已实现指标：

- `doc_count`：rerank 后有效片段数。过低说明召回不足。
- `max_score`：最高语义相似度。过低说明 query 与资料语义距离大。
- `query_coverage`：问题 token 在候选片段中的覆盖比例。过低说明候选缺少关键实体或约束。
- `source_count`：命中来源数。多网页资料集中用于观察召回是否偏向单一来源。

已实现机制：

- 首轮 dense retrieval：标题库和正文库分别召回。
- 轻量 rerank：使用 query embedding 与候选 chunk embedding 的余弦相似度排序。
- 失败判定：当命中数、最高分或 query 覆盖率低于阈值时触发恢复。
- query expansion：自动生成多条语义改写 query。
- 二次召回：扩展 query 再检索，合并、去重、rerank。
- 事件追踪：`events` 返回 `quality_check`、`recall_expansion_start`、`recall_expansion_done`。

建议继续补充：

- 建立 `eval/` 评测集，记录 `question / expected_doc_id / expected_answer / difficulty`。
- 对每次检索输出 `recall@k`、`MRR`、`nDCG`、`coverage`、`source_diversity`。
- 加入 hard negative，评估相似但错误文档是否被错误召回。
- 将阈值从固定值升级为按语料类型动态校准。

预期效果：

- 对“首轮没搜到但资料中存在答案”的样本，query expansion + 二次召回通常可使 recall@5 提升约 `20% - 40%`，最终需要用业务标注集验证。
- `source_count` 和 `query_coverage` 可提前暴露召回偏置，减少模型基于局部证据过度回答。

## 提升回答准确率的技术优化

已实现：

- 资料集级过滤：通过 `tenant_id + corpus_id` 强制限制检索范围，降低跨用户、跨资料污染。
- 标题/正文双库：标题增强主题召回，正文 chunk 支持细粒度证据。
- 多查询扩展：首轮失败后自动补充不同表达方式。
- 引用式 prompt：要求关键事实标注来源编号，例如 `[1]`。
- 低置信度提示：上下文不足时要求明确说明信息有限。
- 检索结果可视化：前端展示 chunk、score、source，便于排查错误答案。

建议下一阶段加入：

- Hybrid Search：Qdrant dense vector + BM25/sparse vector，改善专有名词、编号、错误码、短 query 命中。
- Cross-Encoder Reranker：如 bge-reranker，把 top 50 压缩到 top 5，提高最终上下文质量。
- HyDE：为抽象问题先生成假设答案，再用假设答案检索，提升概念型问题召回。
- Grounding Check：生成后让轻量模型检查答案每个关键断言是否被 chunk 支持。
- Citation Guard：没有来源支持的句子降级或删除。
- Query Router：根据问题类型选择 FAQ、全文检索、摘要、对比、溯源等不同链路。

预期效果：

- Hybrid Search 对包含实体、编号、英文术语的问题通常能明显提升 exact-match 召回。
- Cross-Encoder Reranker 可降低“语义相似但事实不相关”片段进入 prompt 的概率。
- Grounding Check 可降低幻觉率，代价是增加一次模型调用延迟和成本。

## 高并发与高可用部署设计优化

推荐生产架构：

- API Gateway：鉴权、限流、租户注入、请求追踪、灰度发布。
- Frontend CDN：React 静态构建物部署到 CDN/Nginx/Object Storage。
- FastAPI Retrieval Service：无状态部署，多副本水平扩容。
- Ingestion Worker：网页抓取、解析、embedding、写库异步化，使用 Celery/RQ/Arq。
- Queue：Redis Stream、RabbitMQ 或 Kafka，用于入库削峰。
- Redis Cluster：缓存 query expansion、query embedding、retrieval result、generation result。
- Qdrant Cluster：分片、replica、payload index、定期 snapshot。
- Object Storage：保存原文、解析结果、chunk 快照和评测数据。
- Observability：OpenTelemetry + Prometheus + Grafana + Loki，记录 trace、latency、token、cache hit、recall quality。

已落地的工程基础：

- FastAPI 无状态 API。
- `tenant_id / user_id / permission_group` payload 字段与过滤条件。
- `/health` 暴露缓存统计。
- 请求级 `x-request-id` 透传。
- 前后端分离，前端可独立部署。

高可用目标：

- API 多副本：单实例故障不影响服务。
- Qdrant replica：节点故障时检索可继续。
- Worker 异步化：大批量导入不阻塞在线问答。
- 降级策略：reranker 或大模型不可用时退化到 dense retrieval + 基础模型。

## 降低延迟、缓存与成本优化

已实现：

- `query_expansion_cache`：缓存 query expansion，避免相同问题重复调用模型改写。
- `retrieval_cache`：缓存同一租户、同一 corpus、同一问题的检索结果。
- `generation_cache`：缓存同一上下文下的最终答案。
- `/health` 输出缓存 `hits / misses / writes / evictions`。
- 前端静态资源独立部署，避免由 Python 服务渲染页面。

当前缓存为进程内 TTL Cache：

- 优点：零外部依赖，适合开发和单实例部署。
- 限制：多副本之间不共享，进程重启会丢失。
- 生产替换：Redis Cluster，key 使用 `tenant_id + corpus_id + question_hash + prompt_version`。

建议继续优化：

- Query embedding cache：缓存 embedding 向量，减少向量化耗时。
- Batch embedding：导入大量文档时批量向量化。
- Parallel retrieval：title/body/hybrid/expanded queries 并发执行。
- Adaptive topK：简单问题少取 chunk，复杂问题增加 topK。
- Prompt budget control：按 score、coverage、source diversity 动态裁剪上下文。
- Streaming answer：生成接口支持 SSE，降低首 token 等待。
- Model cascade：低复杂度问题走小模型，低置信度或复杂问题升级大模型。

预期效果：

- 热点问题在命中生成缓存时可跳过检索和模型生成，延迟可降到一次 API 往返。
- 命中检索缓存时可省去 Qdrant 查询和 rerank。
- query expansion 缓存可减少失败恢复链路的额外模型调用。
- 前端 CDN 化后，页面加载与后端计算解耦，后端资源专注于 RAG 推理。

## 代码结构

- `frontend/`：React + Vite 前端工作台。
- `api_server.py`：本地 API 启动脚本。
- `blog_rag/api/app.py`：FastAPI 应用入口。
- `blog_rag/api/schemas.py`：API 请求与响应模型。
- `blog_rag/cache.py`：进程内 TTL 缓存。
- `blog_rag/config.py`：模型、Qdrant collection、chunk、检索和评估阈值配置。
- `blog_rag/document_processing.py`：无 UI 依赖的文档解析与 chunk 构建。
- `blog_rag/retrieval_quality.py`：检索质量评估。
- `blog_rag/runtime.py`：运行时配置、模型和 Qdrant 初始化。
- `blog_rag/services/rag_service.py`：工程化入库、检索、生成服务。
- `qdrant_admin.py`：Qdrant collection 检查与清理脚本。

## 下一步

- 将同步 `/v1/corpora/ingest` 改为异步任务，返回 `job_id`。
- 增加 `/v1/corpora/{id}` 管理接口，支持删除、重建、查看统计。
- 引入 Redis，替换进程内缓存。
- 增加 SSE 流式问答接口。
- 建立 `eval/` 自动评测流水线。
- 接入真实 SSO，让 `tenant_id/user_id/permission_group` 由网关注入。
