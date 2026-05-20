import json

import streamlit as st
from langchain_core.messages import HumanMessage

from .agent import create_blog_retrieval_tool, get_graph, run_graph_with_progress, update_conversation_memory
from .config import (
    DEFAULT_MODEL_MAX_TOKENS,
    DEFAULT_MODEL_STREAMING,
    DEFAULT_MODEL_TEMPERATURE,
    MAX_BATCH_URLS,
    MODEL_PROVIDER,
    get_provider_label,
)
from .ingestion import add_extracted_batch_to_qdrant, add_extracted_to_qdrant, extract_from_markdown, extract_from_url
from .models import initialize_components, required_config_ready
from .session_state import init_session_state, record_retrieval_event, switch_chat_session, sync_active_session


def inject_global_styles():
    st.markdown(
        """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: 0; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(24, 32, 48, 0.08); }
  .stButton > button, .stDownloadButton > button { border-radius: 8px; min-height: 2.6rem; }
  .stTextInput input, .stTextArea textarea { border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { font-weight: 650; }
  .rag-hero {
    padding: 1.2rem 1.35rem;
    border: 1px solid rgba(24, 32, 48, 0.08);
    border-radius: 8px;
    background: linear-gradient(135deg, #f8fbff 0%, #f5faf7 55%, #fffdf7 100%);
  }
  .metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .75rem;
    margin-top: .8rem;
  }
  .metric-pill {
    border: 1px solid rgba(24, 32, 48, 0.08);
    border-radius: 8px;
    padding: .65rem .8rem;
    background: rgba(255,255,255,.72);
  }
  .metric-pill small { color: #667085; display:block; }
  .metric-pill strong { color: #182030; font-size: 1rem; }
  .soft-panel {
    border: 1px solid rgba(24, 32, 48, 0.08);
    border-radius: 8px;
    padding: 1rem;
    background: #ffffff;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    title = st.session_state.get("current_blog_title") or "等待导入"
    source = st.session_state.get("last_loaded_url") or "URL / Markdown"
    content_kind = st.session_state.get("current_content_kind") or "-"
    image_count = st.session_state.get("current_image_count", 0)
    document_count = st.session_state.get("current_document_count", 0)
    st.markdown(
        f"""
<div class="rag-hero">
  <div style="font-size:2rem;font-weight:760;color:#162033;">AI Blog Search</div>
  <div style="color:#526071;margin-top:.25rem;">导入网页或 Markdown，连同图片说明一起检索，让 Agent 基于材料回答。</div>
  <div class="metric-strip">
    <div class="metric-pill"><small>当前材料</small><strong>{title}</strong></div>
    <div class="metric-pill"><small>资料数 / 类型</small><strong>{document_count} / {content_kind}</strong></div>
    <div class="metric-pill"><small>图片线索</small><strong>{image_count} 条</strong></div>
  </div>
  <div style="color:#667085;margin-top:.7rem;font-size:.86rem;word-break:break-all;">{source}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.subheader("运行配置")
        st.caption(f"Provider: {get_provider_label()} (`{MODEL_PROVIDER}`)")
        qdrant_host = st.text_input("Qdrant Host", value=st.session_state.qdrant_host, type="password")
        qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")

        gemini_api_key = st.session_state.gemini_api_key
        deepseek_api_key = st.session_state.deepseek_api_key
        if MODEL_PROVIDER == "gemini":
            gemini_api_key = st.text_input("Gemini API Key", value=gemini_api_key, type="password")
            st.info("Gemini 用于聊天和 embedding。")
        else:
            deepseek_api_key = st.text_input("DeepSeek API Key", value=deepseek_api_key, type="password")
            st.info("DeepSeek 负责回答，本地 FastEmbed 负责向量化。")

        if st.button("保存配置", use_container_width=True):
            st.session_state.qdrant_host = qdrant_host
            st.session_state.qdrant_api_key = qdrant_api_key
            st.session_state.gemini_api_key = gemini_api_key
            st.session_state.deepseek_api_key = deepseek_api_key
            st.success("配置已保存")

        st.divider()
        st.subheader("历史会话")
        sessions = st.session_state.get("chat_sessions", {})
        if sessions:
            sorted_sessions = sorted(sessions.values(), key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
            options = [item["id"] for item in sorted_sessions]
            labels = {item["id"]: f'{item.get("title") or "Untitled"} · {item.get("imported_at", "")[:10]}' for item in sorted_sessions}
            current_id = st.session_state.get("active_session_id")
            selected = st.selectbox("切换会话", options=options, index=options.index(current_id) if current_id in options else 0, format_func=lambda value: labels.get(value, value))
            if selected != current_id:
                switch_chat_session(selected)
                st.rerun()
        else:
            st.caption("导入材料后会自动创建会话。")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("清空对话", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_memory = ""
                sync_active_session()
                st.rerun()
        with col_b:
            if st.button("清空调试", use_container_width=True):
                st.session_state.retrieval_debug = []
                st.session_state.retrieval_events = []
                st.rerun()


def render_model_settings():
    with st.popover("模型参数"):
        st.slider("Temperature", 0.0, 1.5, value=float(st.session_state.model_temperature), step=0.05, key="model_temperature")
        st.checkbox("Streaming", value=bool(st.session_state.model_streaming), key="model_streaming")
        st.number_input("Max Tokens", 256, 8192, value=int(st.session_state.model_max_tokens), step=128, key="model_max_tokens")
        if st.button("恢复默认"):
            st.session_state.model_temperature = DEFAULT_MODEL_TEMPERATURE
            st.session_state.model_streaming = DEFAULT_MODEL_STREAMING
            st.session_state.model_max_tokens = DEFAULT_MODEL_MAX_TOKENS
            st.rerun()


def render_chat_history(height: int = 520):
    container = st.container(height=height, border=True)
    with container:
        if not st.session_state.chat_history:
            st.caption("还没有对话。提交第一个问题后，回答会出现在这里。")
            return
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])


def render_ingest_tab(client, title_db, body_db):
    st.markdown("### 导入材料")
    st.caption("支持网页 URL 和 Markdown 文件；页面或 Markdown 中的图片 alt、title、caption、src 会作为图片线索入库。")

    mode = st.radio("导入方式", options=["网页 URL", "Markdown 文件"], horizontal=True)
    if mode == "网页 URL":
        examples = ["https://lilianweng.github.io/posts/2023-06-23-agent/", "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"]
        picked = st.selectbox("示例 URL", options=["不使用", *examples])
        with st.form("url_ingest_form", clear_on_submit=False):
            url_text = st.text_area(
                "材料链接",
                value="" if picked == "不使用" else picked,
                placeholder="每行一个 URL，最多支持批量导入 20 个网页",
                height=140,
            )
            submitted = st.form_submit_button("抓取并入库", use_container_width=True)
        if submitted and url_text.strip():
            urls = []
            for line in url_text.splitlines():
                candidate = line.strip()
                if candidate and candidate not in urls:
                    urls.append(candidate)
            if len(urls) > MAX_BATCH_URLS:
                st.error(f"单次最多导入 {MAX_BATCH_URLS} 个 URL，请拆分批次。")
                return
            with st.status("正在准备导入...", expanded=True) as status:
                try:
                    extracted_items = []
                    for index, url in enumerate(urls, start=1):
                        status.write(f"抓取网页并清洗正文：{index}/{len(urls)}")
                        extracted = extract_from_url(url)
                        status.write(f"发现 {len(extracted.image_notes)} 条图片线索：{extracted.title}")
                        extracted_items.append(extracted)
                    status.write("写入 Qdrant 向量库")
                    ok = add_extracted_batch_to_qdrant(extracted_items, client, title_db, body_db)
                    status.update(label="导入完成" if ok else "导入失败", state="complete" if ok else "error")
                except Exception as exc:
                    status.update(label="导入失败", state="error")
                    st.error(exc)
    else:
        uploaded = st.file_uploader("上传 Markdown", type=["md", "markdown", "txt"])
        if uploaded:
            content = uploaded.getvalue().decode("utf-8", errors="ignore")
            with st.expander("预览", expanded=False):
                st.markdown(content[:4000])
            if st.button("解析并入库", use_container_width=True):
                with st.status("正在解析 Markdown...", expanded=True) as status:
                    extracted = extract_from_markdown(uploaded.name, content)
                    status.write(f"发现 {len(extracted.image_notes)} 条图片线索")
                    status.write("写入 Qdrant 向量库")
                    ok = add_extracted_to_qdrant(extracted, client, title_db, body_db)
                    status.update(label="导入完成" if ok else "导入失败", state="complete" if ok else "error")

    st.divider()
    render_model_settings()


def render_chat_tab(retriever_tool):
    st.markdown("### 对话问答")
    left, right = st.columns([0.64, 0.36], vertical_alignment="top")
    with left:
        render_chat_history()
    with right:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("**当前工作流**")
        st.caption("提交后会依次经历：理解问题、检索材料、必要时改写查询、生成回答。等待时会显示进度，避免用户只看到空白旋转。")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.conversation_memory:
            with st.expander("对话记忆", expanded=False):
                st.text(st.session_state.conversation_memory)

    disabled = not bool(st.session_state.last_loaded_url)
    if disabled:
        st.info("请先导入网页或 Markdown。")
    user_prompt = st.chat_input("针对当前材料提问...", disabled=disabled)
    if user_prompt:
        graph = get_graph(retriever_tool)
        inputs = {"messages": [HumanMessage(content=user_prompt)], "rewrite_count": 0}
        try:
            st.session_state.retrieval_debug = []
            st.session_state.retrieval_events = []
            record_retrieval_event("question_submitted", question=user_prompt, active_blog_id=st.session_state.get("current_blog_id", ""))
            with st.status("正在处理你的问题...", expanded=True) as status:
                response = run_graph_with_progress(graph, inputs, status=status)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": str(response)})
            update_conversation_memory(user_prompt, str(response))
            sync_active_session()
            st.rerun()
        except Exception as exc:
            st.error(f"生成失败：{exc}")


def render_retrieval_visualization():
    quality = st.session_state.get("retrieval_quality", {})
    if quality:
        cols = st.columns(4)
        cols[0].metric("命中片段", quality.get("doc_count", 0))
        cols[1].metric("最高分", round(float(quality.get("max_score", 0.0)), 3))
        cols[2].metric("查询覆盖", round(float(quality.get("query_coverage", 0.0)), 3))
        cols[3].metric("来源数", quality.get("source_count", 0))

    events = st.session_state.get("retrieval_events", [])
    if events:
        with st.expander("检索链路", expanded=True):
            for event in events[-12:]:
                payload = {key: value for key, value in event.items() if key not in {"time", "event"}}
                st.code(f'{event["time"]} | {event["event"]}\n' + json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    if not st.session_state.retrieval_debug:
        st.info("暂无检索数据。先提交一个问题。")
        return

    st.markdown("### 命中片段")
    for item in st.session_state.retrieval_debug:
        st.progress(min(max(item["score"], 0.0), 1.0), text=f'#{item["rank"]} {item["type"]} | score={item["score"]}')
        st.caption(f'{item["title"]}\nsource: {item.get("source", "")}\n\n{item["preview"]}...')


def main():
    st.set_page_config(page_title="AI Blog Search", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    inject_global_styles()
    render_sidebar()
    render_app_header()

    if not required_config_ready():
        st.warning(f"请先在左侧栏配置 Qdrant 和 {get_provider_label()} API Key。")
        return

    try:
        embedding_model, client, title_db, body_db = initialize_components()
    except Exception as exc:
        st.error(f"初始化失败：{exc}")
        return
    if not all([embedding_model, client, title_db, body_db]):
        return

    retriever_tool = create_blog_retrieval_tool(title_db, body_db, embedding_model)
    tabs = st.tabs(["导入", "问答", "检索可视化"])
    with tabs[0]:
        render_ingest_tab(client, title_db, body_db)
    with tabs[1]:
        render_chat_tab(retriever_tool)
    with tabs[2]:
        render_retrieval_visualization()
