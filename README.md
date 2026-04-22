# Agentic RAG — 单篇博客问答系统

本项目实现了一个面向单篇博客网页的 Agentic RAG（Retrieval-Augmented Generation）问答系统。
核心思想是：抓取单个博客页面 → 清洗并分块正文（标题单独处理）→ 向量化并写入 Qdrant → 用户提问时通过检索工具召回候选片段并 Rerank → 用大模型生成严格基于检索上下文的答案。

**主要用途**：对单篇博客或长文进行基于内容的精确问答，支持检索可视化与增量会话记忆。

**文件**：
- 主程序: [AI_blog.py](AI_blog.py)

**快速概览**
- 将页面标题与正文分别作为两个 collection 存入 Qdrant（TITLE_COLLECTION / BODY_COLLECTION）。
- 标题用于主题级召回，正文切块用于细粒度检索。
- 检索后通过轻量级 Rerank（query embedding 与候选文档 embedding 的余弦相似度）排序，标题结果附加权重以提升主题命中率。
- 使用 LangGraph 构建 Agentic 流程：`agent -> retrieve -> grade_documents -> (rewrite | generate)`，支持有限次数的 query 重写以避免死循环。

**核心特性**
- 网页解析与清洗：使用 BeautifulSoup 删除广告、导航、脚本等噪声，并尽量选取 `article` / `main` 等区域作为正文来源。
- 分块与向量化：正文使用基于 tiktoken 的 `RecursiveCharacterTextSplitter` 切块，标题与正文分别做 embedding。
- 双库策略：分别维护标题库与正文库，便于平衡主题与细节召回。
- 二次排序（Rerank）：对初步召回候选做 query 重嵌入并计算余弦相似度，保留 Top-K 给生成器。
- LangGraph Agent：强制模型先调用检索工具，评估检索结果相关性，不相关则改写查询重试（上限由 `MAX_REWRITE_ATTEMPTS` 控制）。
- 会话记忆压缩：对话摘要（conversation memory）以增量方式维护，供后续轮次参考但不占用大量上下文。
- 检索可视化：页面显示 rerank 后的候选、分数、类型与片段预览，便于调试与分析。

**主要模块与关键函数（在 [AI_blog.py](AI_blog.py)）**
- `init_session_state()`：初始化 Streamlit 会话状态。
- `create_embedding_model()`：根据 `MODEL_PROVIDER` 创建 embedding 模型（支持 Gemini 与 DeepSeek+FastEmbed 两种分支）。
- `create_chat_model()`：构建用于生成的聊天模型实例。
- `initialize_components()`：连接 Qdrant，创建 collection 并返回向量库句柄。
- `clean_page_content(raw_html)`：从 HTML 中提取干净的标题与正文。
- `build_documents_from_url(url)`：抓取并构建 title_doc 与 body_docs（正文切块）。
- `create_blog_retrieval_tool(title_db, body_db, current_embedding_model)`：封装检索流程为一个 LangChain 工具（包括标题检索、正文检索、rerank 与格式化）。
- `get_graph(retriever_tool)`：构建 LangGraph 工作流并返回编译后的图对象。
- `generate(state)` / `rewrite(state)` / `agent(state, tools)` / `grade_documents(state)`：LangGraph 各节点实现检索判定、改写与最终生成逻辑。
- `add_documents_to_qdrant(url, client, title_db, body_db)`：将一篇文章写入 Qdrant（写入前会删除同一 URL 的旧数据）。

**配置项（在代码中）**
- `MODEL_PROVIDER`: 选择 `gemini` 或 `deepseek` 分支（默认 `deepseek`）。
- Qdrant: `qdrant_host`, `qdrant_api_key`（在 Streamlit 侧栏输入并保存）。
- 模型 API Key: Gemini 或 DeepSeek 的 API Key（在侧栏输入）。
- 向量维度由 `MODEL_PROVIDER` 决定（Gemini 常为 1536，FastEmbed 示例为 384）。

**运行与使用**
1. 安装依赖（示例）：

```
pip install -r requirements.txt
# 或者至少安装：streamlit beautifulsoup4 qdrant-client langchain langgraph
```

2. 在侧栏填入 `Qdrant Host`（带或不带 scheme，代码会补全 `https://`）与 `Qdrant API key`，并填写相应模型的 API key。
3. 运行：

```
streamlit run AI_blog.py
```

4. 在页面中粘贴博客 URL 并点击 `Enter URL` 导入；导入完成后，在下方输入问题并点击 `Submit Query`。

**实现细节与设计取舍**
- 标题与正文分库：提升主题召回与精确上下文的兼容性，便于不同粒度的检索策略。
- 轻量级 Rerank：避免每次都调用大型 reranker，而是用当前 embedding 模型在候选集上做快速重排，权衡准确性与成本。
- 强制工具调用的 agent：通过 system prompt 强约束模型必须基于检索内容回答，降低“凭常识臆造答案”的风险。
- Query 改写上限：设置 `MAX_REWRITE_ATTEMPTS` 防止 LangGraph 在低相关场景下反复循环。

**常见问题与排障**
- 如果缺少依赖，运行时会抛出 ImportError：请按提示安装相应包（如 `langchain_community`、`langchain_openai`、`fastembed` 等）。
- Qdrant 连接失败：确认 `qdrant_host` 包含主机名或 URL，API Key 有效且 Qdrant 服务可达。
- 重复导入：系统在导入前会删除同一 URL 的旧向量。

**扩展建议**
- 支持多篇文章的批量导入与索引管理。
- 使用更强的 reranker（交互式 cross-encoder）以提升精度。
- 添加基于用户偏好的检索重排序（如偏好技术深度或概览）。
- 将客户端向量化替换为更高效的批量向量化流程以提升吞吐。

如果你希望我把本 README 进一步写成 `requirements.txt`、部署脚本或把 README 转为英文版本，我可以继续完成这些任务。
