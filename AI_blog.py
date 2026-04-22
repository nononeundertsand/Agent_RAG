"""
这个文件实现了一个“面向单篇博客网页的 Agentic RAG 问答系统”，核心能力包括：

1. 用户输入一个博客 URL，系统抓取网页内容。
2. 自动提取页面标题和正文，并尽量移除广告、导航、订阅等噪声。
3. 将“标题”和“正文分块”分别做 embedding，并存入 Qdrant。
4. 用户提问后，Agent 先调用检索工具，再根据检索结果回答。
5. 检索阶段加入了 rerank，以提升命中质量。
6. Query 重写设置了上限，避免 LangGraph 在低相关场景下反复循环。
7. 页面中会展示检索命中的可视化结果，便于调试和理解检索过程。

你可以把这个项目理解为：
“用 Streamlit 做界面，用 LangChain/LangGraph 编排流程，用 Qdrant 存向量，用大模型做判断和生成。”
"""

from functools import partial
from typing import Annotated, Optional, Sequence
from uuid import uuid4

import streamlit as st
from bs4 import BeautifulSoup
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointIdsList, VectorParams
from typing_extensions import TypedDict

# 这里用于切换整个项目当前使用的“大模型提供商”。
# 目前支持两种模式：
# - gemini: 聊天和 embedding 都使用 Google Gemini 相关能力
# - deepseek: 聊天使用 DeepSeek，embedding 使用本地 FastEmbed
MODEL_PROVIDER = "deepseek"

# Gemini 的默认聊天模型和 embedding 模型名称。
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# DeepSeek 的默认聊天模型和接口地址。
# DeepSeek 这里只负责“对话生成”，不负责向量化。
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 当使用 DeepSeek 分支时，本地使用的 embedding 模型。
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# 为了做到“标题单独检索、正文单独检索”，这里把一个逻辑库拆成两个 collection。
# - title collection: 只存标题
# - body collection: 存正文切块
QDRANT_COLLECTION_PREFIX = "qdrant_db"
TITLE_COLLECTION_NAME = f"{QDRANT_COLLECTION_PREFIX}_title"
BODY_COLLECTION_NAME = f"{QDRANT_COLLECTION_PREFIX}_body"

# Query 改写最多重试多少次，超过后直接用当前最佳结果生成答案，避免死循环。
MAX_REWRITE_ATTEMPTS = 2

# 初次召回时，标题和正文分别取多少条候选。
BODY_RETRIEVAL_K = 8
TITLE_RETRIEVAL_K = 3

# rerank 之后最终保留多少条文档片段给大模型。
RERANK_TOP_K = 5


# Streamlit 页面基础配置。
st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")


def init_session_state():
    """
    初始化 Streamlit 的 session_state。

    Streamlit 每次交互都会触发脚本重跑，因此一些“需要在多次交互间保留”的值
    要放进 st.session_state 中，例如：
    - 用户输入的 API Key
    - 最近一次检索结果的可视化数据
    - 最近成功载入的 URL
    """
    defaults = {
        "qdrant_host": "",
        "qdrant_api_key": "",
        "gemini_api_key": "",
        "deepseek_api_key": "",
        "retrieval_debug": [],
        "last_loaded_url": "",
        "chat_history": [],
        "conversation_memory": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_provider_label() -> str:
    """
    根据 MODEL_PROVIDER 返回更适合页面展示的人类可读名称。
    """
    return "Gemini" if MODEL_PROVIDER == "gemini" else "DeepSeek"


def create_embedding_model():
    """
    创建 embedding 模型对象。

    设计思路：
    - 如果当前模式是 Gemini，则直接使用 Gemini 官方 embedding 接口。
    - 如果当前模式是 DeepSeek，则聊天走 DeepSeek，但 embedding 走本地 FastEmbed。

    这么做的好处是：
    DeepSeek 分支不依赖额外的在线 embedding API，也能完成向量检索。
    """
    if MODEL_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=st.session_state.gemini_api_key,
        )

    if MODEL_PROVIDER == "deepseek":
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
        except ImportError as exc:
            raise ImportError(
                "DeepSeek 模式需要安装 langchain-community 与 fastembed。"
            ) from exc

        return FastEmbedEmbeddings(
            model_name=FASTEMBED_MODEL,
            cache_dir="./model_cache",
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def create_chat_model():
    """
    创建聊天模型对象。

    这里返回的是“真正负责推理和生成答案”的大模型：
    - Gemini 模式：使用 ChatGoogleGenerativeAI
    - DeepSeek 模式：通过 OpenAI 兼容接口包装为 ChatOpenAI
    """
    if MODEL_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=st.session_state.gemini_api_key,
            temperature=0,
            streaming=True,
            model=GEMINI_CHAT_MODEL,
        )

    if MODEL_PROVIDER == "deepseek":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "DeepSeek 模式需要安装 langchain-openai。"
            ) from exc

        return ChatOpenAI(
            api_key=st.session_state.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_CHAT_MODEL,
            temperature=0,
            streaming=True,
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def required_config_ready() -> bool:
    """
    检查当前运行所需的最小配置是否齐全。

    所有模式都必须有：
    - qdrant_host
    - qdrant_api_key

    额外再根据当前提供商检查对应的模型 API Key。
    """
    common_ready = all([
        st.session_state.qdrant_host,
        st.session_state.qdrant_api_key,
    ])
    if not common_ready:
        return False

    if MODEL_PROVIDER == "gemini":
        return bool(st.session_state.gemini_api_key)

    if MODEL_PROVIDER == "deepseek":
        return bool(st.session_state.deepseek_api_key)

    return False


def set_sidebar():
    """
    渲染左侧配置栏，让用户输入：
    - Qdrant 地址
    - Qdrant API Key
    - 当前模型供应商对应的 API Key

    用户点击 Done 后，配置会写入 session_state。
    """
    with st.sidebar:
        st.subheader("API Configuration")
        st.caption(f"Current provider: {get_provider_label()} (`MODEL_PROVIDER = \"{MODEL_PROVIDER}\"`)")

        qdrant_host = st.text_input(
            "Enter your Qdrant Host URL:",
            value=st.session_state.qdrant_host,
            type="password",
        )
        qdrant_api_key = st.text_input(
            "Enter your Qdrant API key:",
            value=st.session_state.qdrant_api_key,
            type="password",
        )

        gemini_api_key = st.session_state.gemini_api_key
        deepseek_api_key = st.session_state.deepseek_api_key

        if MODEL_PROVIDER == "gemini":
            gemini_api_key = st.text_input(
                "Enter your Gemini API key:",
                value=st.session_state.gemini_api_key,
                type="password",
            )
            st.info("当前使用 Gemini 作为聊天模型和 embedding 模型。")
        elif MODEL_PROVIDER == "deepseek":
            deepseek_api_key = st.text_input(
                "Enter your DeepSeek API key:",
                value=st.session_state.deepseek_api_key,
                type="password",
            )
            st.info("当前使用 DeepSeek 负责回答生成，FastEmbed 负责本地向量化。")
        else:
            st.error(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

        if st.button("Done"):
            if not qdrant_host or not qdrant_api_key:
                st.warning("Please fill in Qdrant host and Qdrant API key.")
                return

            if MODEL_PROVIDER == "gemini" and not gemini_api_key:
                st.warning("Please fill in the Gemini API key.")
                return

            if MODEL_PROVIDER == "deepseek" and not deepseek_api_key:
                st.warning("Please fill in the DeepSeek API key.")
                return

            st.session_state.qdrant_host = qdrant_host
            st.session_state.qdrant_api_key = qdrant_api_key
            st.session_state.gemini_api_key = gemini_api_key
            st.session_state.deepseek_api_key = deepseek_api_key
            st.success("API keys saved!")


def initialize_components():
    """
    初始化系统运行所需的核心组件：
    - embedding 模型
    - Qdrant 客户端
    - 标题向量库
    - 正文向量库

    注意这里没有使用 recreate_collection，而是“如果不存在才创建”。
    这样可以避免 Streamlit 每次重跑脚本时把之前已导入的数据清空。
    """
    if not required_config_ready():
        return None, None, None, None

    try:
        current_embedding_model = create_embedding_model()

        # 有些用户可能只填了域名，没有带 http/https。
        # 这里自动补成 https，减少配置出错概率。
        host = st.session_state.qdrant_host
        if not host.startswith("http"):
            host = "https://" + host

        client = QdrantClient(
            url=host,
            api_key=st.session_state.qdrant_api_key,
            check_compatibility=False,
        )

        # 不同 embedding 模型的向量维度不同。
        # Gemini embedding-001 常见为 1536 维，FastEmbed 这里是 384 维。
        dim = 1536 if MODEL_PROVIDER == "gemini" else 384

        # 先读取当前 Qdrant 中已有的 collection，避免重复创建时报错。
        existing_collections = {
            collection.name for collection in client.get_collections().collections
        }

        for collection_name in [TITLE_COLLECTION_NAME, BODY_COLLECTION_NAME]:
            if collection_name not in existing_collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    ),
                )

            # Qdrant 对 payload 过滤字段通常要求先建索引。
            # 后面我们会按 metadata.source 过滤并删除同一 URL 的旧数据，
            # 因此这里提前为该字段创建 keyword 索引。
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.source",
                    field_schema="keyword",
                )
            except Exception:
                # 如果索引已存在，或当前 Qdrant 版本返回“重复创建”类异常，这里直接忽略。
                pass

        # 分别构建两个向量库对象：
        # 1. 标题库：适合召回更高层语义/主题级信息
        # 2. 正文库：适合召回细粒度内容片段
        title_db = QdrantVectorStore(
            client=client,
            collection_name=TITLE_COLLECTION_NAME,
            embedding=current_embedding_model,
        )
        body_db = QdrantVectorStore(
            client=client,
            collection_name=BODY_COLLECTION_NAME,
            embedding=current_embedding_model,
        )
        return current_embedding_model, client, title_db, body_db
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None, None, None


class AgentState(TypedDict):
    """
    LangGraph 中流转的状态对象。

    字段说明：
    - messages:
      整个图中所有节点共享的消息序列，LangGraph 会借助 add_messages 自动拼接。
    - rewrite_count:
      当前 query 被改写了多少次，用来阻止无限重试。
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int


def grade_documents(state):
    """
    判断“当前检索到的文档”是否足够回答用户问题。

    返回值不是布尔值，而是 LangGraph 中的“下一跳节点名”：
    - "generate": 文档相关，直接生成答案
    - "rewrite": 文档不够相关，先改写问题再检索一次

    同时加入了防死循环逻辑：
    如果 rewrite_count 已达到上限，就算相关性一般，也不再继续改写，
    直接进入 generate 阶段，用当前最好的检索结果回答。
    """
    print("---CHECK RELEVANCE---")

    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    rewrite_count = state.get("rewrite_count", 0)

    prompt = f"""
You are a grader assessing relevance of a retrieved document to a user question.

Document:
{docs}

Question:
{question}

Answer ONLY with "yes" or "no".
"""

    response = create_chat_model().invoke(prompt)
    score = response.content.lower().strip()

    if "yes" in score:
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    if rewrite_count >= MAX_REWRITE_ATTEMPTS:
        print("---DECISION: MAX REWRITE REACHED, GENERATE WITH BEST DOCS---")
        return "generate"

    print("---DECISION: DOCS NOT RELEVANT---")
    return "rewrite"


def agent(state, tools):
    """
    Agent 节点：负责决定是否调用工具。

    这里通过 system prompt 强约束模型：
    必须先调用检索工具，不能直接凭自己的先验知识回答。

    这样做的目标是让整个问答系统真正“基于网页内容回答”，
    而不是让模型自由发挥。
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    memory_summary = st.session_state.get("conversation_memory", "")
    recent_history = get_recent_chat_history_text()

    memory_block = ""
    if memory_summary:
        memory_block += f"\nConversation memory summary:\n{memory_summary}\n"
    if recent_history:
        memory_block += f"\nRecent chat history:\n{recent_history}\n"

    system_prompt = SystemMessage(
        content="""
You are an AI assistant that MUST use the provided retrieval tool to answer any question.

Rules:
1. ALWAYS call the retriever tool before answering
2. DO NOT answer from your own knowledge
3. DO NOT ask user for more information
4. Use the tool even if the question seems simple
5. Base your final answer ONLY on retrieved content
"""
        + memory_block
    )

    new_messages = [system_prompt] + list(messages)
    model = create_chat_model().bind_tools(tools)
    response = model.invoke(new_messages)
    return {"messages": [response]}


def rewrite(state):
    """
    当检索结果不够相关时，对用户问题做一次“查询改写”。

    例如：
    用户问题表述较口语、较模糊，或者没有使用文中常见术语时，
    改写后的问题更适合用来做向量检索。

    同时这里会把 rewrite_count + 1。
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    memory_summary = st.session_state.get("conversation_memory", "")
    recent_history = get_recent_chat_history_text()

    msg = [
        HumanMessage(
            content=f"""
Look at the input and reason about the underlying semantic intent.

Conversation memory summary:
{memory_summary or "None"}

Recent chat history:
{recent_history or "None"}

Here is the initial question:
-------
{question}
-------

Formulate an improved question for retrieval:
""",
        )
    ]

    response = create_chat_model().invoke(msg)
    return {
        "messages": [response],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def generate(state):
    """
    最终生成答案的节点。

    这一步拿到：
    - 原始用户问题
    - 检索返回的上下文

    再套用 LangChain Hub 上的 RAG prompt 生成最终答案。
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    memory_summary = st.session_state.get("conversation_memory", "")
    recent_history = get_recent_chat_history_text()

    prompt_template = hub.pull("rlm/rag-prompt")
    output_parser = StrOutputParser()
    rag_chain = prompt_template | create_chat_model() | output_parser

    augmented_question = question
    if memory_summary:
        augmented_question += f"\n\nConversation memory summary:\n{memory_summary}"
    if recent_history:
        augmented_question += f"\n\nRecent chat history:\n{recent_history}"

    response = rag_chain.invoke({"context": docs, "question": augmented_question})
    return {"messages": [response]}


def get_recent_chat_history_text(limit: int = 6) -> str:
    """
    将最近若干轮对话整理成纯文本，供模型在当前轮次参考。

    这里保留原始对话片段，适合处理“上一问里提到的它/这个/前面那个概念”之类的指代。
    """
    history = st.session_state.get("chat_history", [])
    if not history:
        return ""

    recent_items = history[-limit:]
    lines = []
    for item in recent_items:
        role = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def update_conversation_memory(user_query: str, assistant_response: str):
    """
    将“已有记忆 + 最新一轮问答”压缩成新的记忆摘要。

    这是一个增量记忆策略：
    - 不把所有历史对话原样无限拼接给模型
    - 而是持续维护一份更短的摘要

    这样可以节省上下文长度，同时保留跨轮对话的关键事实、术语和用户关注点。
    """
    previous_memory = st.session_state.get("conversation_memory", "")
    prompt = f"""
You are maintaining compressed conversational memory for a blog QA assistant.

Update the memory summary using:
1. Existing memory summary
2. The latest user question
3. The latest assistant answer

Keep only durable, useful information for future turns:
- user intent and ongoing subtopics
- resolved references and terminology
- important facts already established in the conversation
- any explicit user preferences

Do not include unnecessary wording.
Return a concise memory summary in Chinese bullet-style plain text.

Existing memory summary:
{previous_memory or "None"}

Latest user question:
{user_query}

Latest assistant answer:
{assistant_response}
"""
    try:
        memory_response = create_chat_model().invoke(prompt)
        st.session_state.conversation_memory = memory_response.content.strip()
    except Exception:
        fallback_lines = []
        if previous_memory:
            fallback_lines.append(previous_memory)
        fallback_lines.append(f"用户最近问题: {user_query}")
        fallback_lines.append(f"最近回答摘要: {assistant_response[:300]}")
        st.session_state.conversation_memory = "\n".join(fallback_lines[-6:])


def render_chat_history():
    """
    在页面上渲染可滚动的历史对话区域。

    使用固定高度的 container，可以在历史对话较多时滚动查看。
    """
    st.markdown("### Chat History")
    history_container = st.container(height=420, border=True)
    with history_container:
        if not st.session_state.chat_history:
            st.caption("当前还没有历史对话。提交第一个问题后，这里会显示完整聊天记录。")
            return

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

    if st.session_state.conversation_memory:
        with st.expander("Conversation Memory Summary", expanded=False):
            st.text(st.session_state.conversation_memory)


def cosine_similarity(vec1, vec2):
    """
    手动计算两个向量的余弦相似度。

    这里单独写一个函数，是因为后面的 rerank 需要对“query 向量”和“候选文档向量”
    做再次打分。
    """
    numerator = sum(x * y for x, y in zip(vec1, vec2))
    denominator = (sum(x * x for x in vec1) ** 0.5) * (sum(y * y for y in vec2) ** 0.5)
    if not denominator:
        return 0.0
    return numerator / denominator


def clean_page_content(raw_html: str) -> tuple[str, str]:
    """
    从原始 HTML 中抽取“较干净的标题和正文”。

    处理步骤：
    1. 用 BeautifulSoup 解析 HTML
    2. 删除 script/style/nav/footer/aside 等明显无用区域
    3. 删除 class 名中含 ad、banner、cookie、sidebar 等关键词的块
    4. 优先从 article/main 等区域提取正文
    5. 对正文逐行清洗，去掉过短文本和常见噪声行

    返回：
    - title: 页面标题
    - body: 清洗后的正文纯文本
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # 先删除结构性噪声标签。
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    # 再删除 class 名看起来像广告、弹窗、侧边栏的区域。
    for tag in soup.find_all(
        attrs={
            "class": lambda value: value and any(
                keyword in " ".join(value).lower()
                for keyword in ["ad", "ads", "banner", "popup", "subscribe", "cookie", "promo", "related", "sidebar"]
            )
        }
    ):
        tag.decompose()

    # 标题优先取 <title>，没有的话再尝试取 <h1>。
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else ""

    # 尝试从更可能是“正文”的区域中选出文字最多的一块。
    main_candidates = []
    for selector in ["article", "main", "[role='main']", ".post-content", ".entry-content", ".article-content"]:
        main_candidates.extend(soup.select(selector))

    target = max(
        main_candidates,
        key=lambda node: len(node.get_text(" ", strip=True)),
        default=soup.body or soup,
    )

    body_text = target.get_text("\n", strip=True)

    # 行级清洗：
    # - 太短的行通常没信息量，去掉
    # - 含 advertisement / subscribe / cookie 等词的行当作噪声去掉
    cleaned_lines = []
    for line in body_text.splitlines():
        normalized = " ".join(line.split())
        if len(normalized) < 20:
            continue
        lower_line = normalized.lower()
        if any(keyword in lower_line for keyword in ["advertisement", "subscribe", "all rights reserved", "cookie"]):
            continue
        cleaned_lines.append(normalized)

    return title, "\n".join(cleaned_lines)


def build_documents_from_url(url: str) -> tuple[Optional[Document], list[Document]]:
    """
    给定 URL，构造可写入向量库的 LangChain Document。

    返回两个部分：
    - title_doc: 只包含标题的 Document
    - body_docs: 正文分块后的多个 Document

    这样后续就可以实现“标题和正文分开 embedding / 分开召回”。
    """
    docs = WebBaseLoader(url).load()
    if not docs:
        return None, []

    source_doc = docs[0]
    title, cleaned_body = clean_page_content(source_doc.page_content)

    if not cleaned_body.strip():
        return None, []

    # metadata 会跟随文档一起写入向量库，后续可用于过滤、展示和调试。
    metadata = {
        **source_doc.metadata,
        "source": url,
    }

    title_doc = Document(
        page_content=title or metadata.get("title", url),
        metadata={**metadata, "content_type": "title", "page_title": title or url},
    )

    # 正文会切块，因为一整篇文章通常太长，不适合直接作为一个 embedding 单元。
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )
    body_docs = text_splitter.create_documents(
        [cleaned_body],
        metadatas=[{**metadata, "content_type": "body", "page_title": title_doc.metadata["page_title"]}],
    )
    return title_doc, body_docs


def rerank_documents(query: str, current_embedding_model, title_docs: list[Document], body_docs: list[Document]):
    """
    对初步召回结果做二次排序（rerank）。

    这里采用的是一个“轻量做法”：
    1. 对 query 重新做 embedding
    2. 对每条候选文档内容也做 embedding
    3. 用余弦相似度重新计算分数
    4. 标题结果乘以 1.2 权重，让主题命中更容易排前面

    这么做的意义是：
    初次召回只是粗筛，rerank 相当于在粗筛结果上再排一次序，通常会更稳。

    同时函数还会把 TopK 结果写入 session_state，用于页面上的“查询可视化”。
    """
    query_vector = current_embedding_model.embed_query(query)
    scored_docs = []

    # 标题通常更浓缩主题，因此人为加一点权重。
    for doc in title_docs:
        doc_vector = current_embedding_model.embed_query(doc.page_content)
        score = cosine_similarity(query_vector, doc_vector) * 1.2
        scored_docs.append((score, "title", doc))

    for doc in body_docs:
        doc_vector = current_embedding_model.embed_query(doc.page_content)
        score = cosine_similarity(query_vector, doc_vector)
        scored_docs.append((score, "body", doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    top_items = scored_docs[:RERANK_TOP_K]

    st.session_state.retrieval_debug = [
        {
            "rank": index,
            "score": round(score, 4),
            "type": doc_type,
            "title": doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled",
            "preview": doc.page_content[:220],
        }
        for index, (score, doc_type, doc) in enumerate(top_items, start=1)
    ]

    return [doc for _, _, doc in top_items]


def format_docs_for_prompt(docs: list[Document]) -> str:
    """
    把多个 Document 拼成一个适合塞进 prompt 的大字符串。

    拼接时保留：
    - 排名编号
    - 内容类型（title / body）
    - 页面标题
    - 正文内容

    这样模型在生成答案时更容易理解“上下文来自哪里”。
    """
    sections = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled"
        content_type = doc.metadata.get("content_type", "body")
        sections.append(f"[{index}] ({content_type}) {title}\n{doc.page_content}")
    return "\n\n".join(sections)


def create_blog_retrieval_tool(title_db, body_db, current_embedding_model):
    """
    把“标题检索 + 正文检索 + rerank”封装成一个 LangChain Tool。

    这样 LangGraph 中的 agent 节点就可以像调用普通工具一样调用它。

    整个检索流程是：
    1. 标题库召回 title_docs
    2. 正文库召回 body_docs
    3. rerank_documents 统一重排
    4. format_docs_for_prompt 输出给大模型
    """
    @tool("retrieve_blog_posts")
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about the indexed blog post."""
        title_docs = title_db.similarity_search(query, k=TITLE_RETRIEVAL_K)
        body_docs = body_db.similarity_search(query, k=BODY_RETRIEVAL_K)
        reranked_docs = rerank_documents(query, current_embedding_model, title_docs, body_docs)
        return format_docs_for_prompt(reranked_docs)

    return retrieve_blog_posts


def get_graph(retriever_tool):
    """
    构建 LangGraph 工作流。

    当前图的结构如下：

    START
      -> agent
      -> 如果模型决定要调用工具，则进入 retrieve
      -> retrieve 后通过 grade_documents 判断：
           - 相关 -> generate
           - 不相关 -> rewrite
      -> rewrite 后回到 agent 再试一次
      -> generate 后结束

    这就是一个典型的“检索 -> 评估 -> 改写 -> 再检索 -> 生成”的 Agentic RAG 流程。
    """
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", partial(agent, tools=tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()


def generate_message(graph, inputs):
    """
    执行整个图，并从流式输出中提取 generate 节点生成的最终答案。

    graph.stream(inputs) 会不断返回图中各节点的中间输出。
    我们只关心 generate 节点的产物，因此这里做了一层筛选。
    """
    generated_message = ""

    for output in graph.stream(inputs):
        print(f"Graph output: {output}")
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]

    return generated_message


def clear_documents_by_url(client, collection_name, url):
    """
    删除某个 URL 在指定 collection 中旧的数据。

    这样同一篇文章重复导入时，不会在向量库里堆出重复内容。
    """
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=url),
                )
            ]
        ),
        with_payload=False,
        with_vectors=False,
        limit=1000,
    )

    point_ids = [point.id for point in points if point.id is not None]
    if point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
        )


def add_documents_to_qdrant(url, client, title_db, body_db):
    """
    将一篇博客写入 Qdrant。

    处理流程：
    1. 解析 URL，得到 title_doc 和 body_docs
    2. 删除该 URL 的旧向量数据
    3. 标题写入标题库
    4. 正文分块写入正文库
    5. 更新 session_state 中的最近载入 URL
    """
    try:
        title_doc, body_docs = build_documents_from_url(url)
        if not title_doc or not body_docs:
            st.error("No readable article content was extracted from the URL.")
            return False

        clear_documents_by_url(client, TITLE_COLLECTION_NAME, url)
        clear_documents_by_url(client, BODY_COLLECTION_NAME, url)

        title_db.add_documents([title_doc], ids=[str(uuid4())])
        body_db.add_documents(body_docs, ids=[str(uuid4()) for _ in body_docs])

        st.session_state.last_loaded_url = url
        st.session_state.retrieval_debug = []
        st.session_state.chat_history = []
        st.session_state.conversation_memory = ""
        print("title:", title_doc.page_content)
        print("body chunks:", len(body_docs))
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False


def render_retrieval_visualization():
    """
    在页面上展示 rerank 后的检索结果。

    展示内容包括：
    - 排名
    - 分数
    - 类型（标题 / 正文）
    - 片段预览

    这部分主要用于帮助你理解：
    “模型在回答问题前，到底检索到了哪些内容”
    """
    if not st.session_state.retrieval_debug:
        return

    st.markdown("### Query Visualization")
    for item in st.session_state.retrieval_debug:
        progress_value = min(max(item["score"], 0.0), 1.0)
        st.progress(progress_value, text=f'#{item["rank"]} {item["type"]} | score={item["score"]}')
        st.caption(f'{item["title"]}\n\n{item["preview"]}...')


def main():
    """
    程序主入口。

    这个函数把整个页面和业务流程串起来：
    1. 初始化 session state
    2. 渲染侧边栏配置
    3. 初始化 embedding / Qdrant / 向量库
    4. 让用户输入博客 URL 并入库
    5. 让用户提问
    6. 运行 LangGraph 问答流程
    7. 展示最终答案和检索可视化
    """
    init_session_state()
    set_sidebar()

    if not required_config_ready():
        st.warning(f"Please configure the required keys for {get_provider_label()} in the sidebar first.")
        return

    current_embedding_model, client, title_db, body_db = initialize_components()
    if not all([current_embedding_model, client, title_db, body_db]):
        return

    retriever_tool = create_blog_retrieval_tool(title_db, body_db, current_embedding_model)

    # 第一步：让用户输入博客地址，并把网页内容入库。
    url = st.text_input(
        ":link: Paste the blog link:",
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/",
    )
    if st.button("Enter URL"):
        if url:
            with st.spinner("Processing documents..."):
                if add_documents_to_qdrant(url, client, title_db, body_db):
                    st.success("Documents added successfully!")
                else:
                    st.error("Failed to add documents")
        else:
            st.warning("Please enter a URL")

    render_chat_history()

    # 第二步：基于已经导入的博客内容，进行问答。
    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?",
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query")
            return

        # 防止用户没有导入文章就直接提问。
        if not st.session_state.last_loaded_url:
            st.warning("Please load a blog URL first.")
            return

        # LangGraph 初始输入：
        # - messages: 用户原始问题
        # - rewrite_count: 从 0 开始计数
        inputs = {
            "messages": [HumanMessage(content=query)],
            "rewrite_count": 0,
        }
        with st.spinner(f"Generating response with {get_provider_label()}..."):
            try:
                st.session_state.retrieval_debug = []
                response = generate_message(graph, inputs)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": query,
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": str(response),
                })
                update_conversation_memory(query, str(response))
                st.write(response)
                render_retrieval_visualization()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    st.markdown("---")
    st.write(
        f"Current provider: :blue-background[{get_provider_label()}] | Built with :blue-background[LangChain] | :blue-background[LangGraph]"
    )


if __name__ == "__main__":
    main()
