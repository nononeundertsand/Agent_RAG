from functools import partial
from typing import Annotated, Literal, Sequence
from uuid import uuid4

import streamlit as st
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from typing_extensions import TypedDict

from fastembed import TextEmbedding

# 指定一个不会被系统自动清理的目录，例如项目目录下的 model_cache
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir="./model_cache"  # 使用项目目录，避免被系统清理
)


# 手动切换当前使用的大模型提供商。
# 可选值：
# - "gemini"
# - "deepseek"
MODEL_PROVIDER = "deepseek"

# Gemini 默认模型
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# DeepSeek 默认模型
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek 分支下用于向量化的本地 Embedding 模型。
# 原因：DeepSeek 主要提供对话模型接口，这个项目仍然需要 embedding 才能做向量检索。
# FastEmbed 首次运行时通常会下载模型文件。
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Qdrant 集合名
QDRANT_COLLECTION_NAME = "qdrant_db"


st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")


def init_session_state():
    """初始化页面运行过程中需要持久化的配置项。"""
    defaults = {
        "qdrant_host": "",
        "qdrant_api_key": "",
        "gemini_api_key": "",
        "deepseek_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_provider_label() -> str:
    """返回当前配置的人类可读提供商名称。"""
    return "Gemini" if MODEL_PROVIDER == "gemini" else "DeepSeek"


def create_embedding_model():
    """
    根据当前提供商创建 embedding 模型。

    注意：
    - Gemini 分支：直接使用 Gemini Embeddings
    - DeepSeek 分支：使用本地 FastEmbed，保证 RAG 检索链仍然可用
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
            cache_dir="./model_cache"   
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def create_chat_model():
    """根据当前提供商创建聊天模型。"""
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

        # DeepSeek 提供 OpenAI 兼容接口，因此这里使用 ChatOpenAI 适配。
        return ChatOpenAI(
            api_key=st.session_state.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_CHAT_MODEL,
            temperature=0,
            streaming=True,
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def required_config_ready() -> bool:
    """检查当前提供商所需的最小配置是否齐全。"""
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
    """在侧边栏中填写当前运行所需的配置。"""
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
            st.info("当前会使用 Gemini 作为聊天模型和向量化模型。")
        elif MODEL_PROVIDER == "deepseek":
            deepseek_api_key = st.text_input(
                "Enter your DeepSeek API key:",
                value=st.session_state.deepseek_api_key,
                type="password",
            )
            st.info("当前会使用 DeepSeek 负责对话生成，FastEmbed 负责本地向量化。")
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
    if not required_config_ready():
        return None, None, None

    try:
        embedding_model = create_embedding_model()

        # ✅ 修正 host（防止没写 https）
        host = st.session_state.qdrant_host
        if not host.startswith("http"):
            host = "https://" + host

        client = QdrantClient(
            url=host,
            api_key=st.session_state.qdrant_api_key,
            check_compatibility=False
        )

        # ✅ 关键：根据 embedding 自动决定维度
        dim = 1536 if MODEL_PROVIDER == "gemini" else 384

        # ✅ 强制重建 collection（真正正确的位置）
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            ),
        )

        db = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embedding_model,
        )

        return embedding_model, client, db

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None, None


class AgentState(TypedDict):
    # LangGraph 中的状态对象。
    # 整个流程在图节点间传递的核心数据就是 messages。
    messages: Annotated[Sequence[BaseMessage], add_messages]


# def grade_documents(state) -> Literal["generate", "rewrite"]:
#     """
#     判断当前检索到的文档是否和用户问题足够相关。
#     相关则进入生成，不相关则重写问题后重试。
#     """
#     print("---CHECK RELEVANCE---")

#     class Grade(BaseModel):
#         """文档相关性检查结果。"""

#         binary_score: str = Field(description="Relevance score 'yes' or 'no'")

#     llm_with_tool = create_chat_model().with_structured_output(Grade)

#     prompt = PromptTemplate(
#         template="""You are a grader assessing relevance of a retrieved document to a user question. \n
#         Here is the retrieved document: \n\n {context} \n\n
#         Here is the user question: {question} \n
#         If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
#         input_variables=["context", "question"],
#     )

#     chain = prompt | llm_with_tool

#     messages = state["messages"]
#     question = messages[0].content
#     docs = messages[-1].content

#     scored_result = chain.invoke({"question": question, "context": docs})
#     score = scored_result.binary_score

#     if score == "yes":
#         print("---DECISION: DOCS RELEVANT---")
#         return "generate"

#     print("---DECISION: DOCS NOT RELEVANT---")
#     return "rewrite"

def grade_documents(state):
    print("---CHECK RELEVANCE---")

    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content

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

    print("---DECISION: DOCS NOT RELEVANT---")
    return "rewrite"


# def agent(state, tools):
#     """
#     Agent 节点负责决定是否要调用检索工具。
#     如果模型认为需要查资料，就会触发 retriever tool。
#     """
#     print("---CALL AGENT---")
#     messages = state["messages"]

#     model = create_chat_model().bind_tools(tools)
#     response = model.invoke(messages)
#     return {"messages": [response]}

def agent(state, tools):
    print("---CALL AGENT---")
    messages = state["messages"]

    # 🔥 强制工具调用的 system prompt
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
    )

    # 👉 把 system prompt 插到最前面
    new_messages = [system_prompt] + list(messages)

    model = create_chat_model().bind_tools(tools)

    response = model.invoke(new_messages)

    return {"messages": [response]}


def rewrite(state):
    """
    当检索结果不够相关时，先把问题改写得更适合检索。
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n
                    Here is the initial question:
                    \n ------- \n
                    {question}
                    \n ------- \n
                    Formulate an improved question: """,
        )
    ]

    response = create_chat_model().invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    标准 RAG 生成阶段：
    把“用户问题 + 检索到的上下文”一起交给大模型生成最终答案。
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content

    prompt_template = hub.pull("rlm/rag-prompt")
    output_parser = StrOutputParser()
    rag_chain = prompt_template | create_chat_model() | output_parser

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def get_graph(retriever_tool):
    """构建整个 LangGraph 工作流。"""
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
    """从图执行流里提取 generate 节点产出的最终答案。"""
    generated_message = ""

    for output in graph.stream(inputs):
        print(f"Graph output: {output}")
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]

    return generated_message


def add_documents_to_qdrant(url, db):
    """抓取网页内容、切分文本并写入 Qdrant。"""
    try:
        docs = WebBaseLoader(url).load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=100,
        )
        doc_chunks = text_splitter.split_documents(docs)

        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        db.add_documents(documents=doc_chunks, ids=uuids)
        print("docs:", docs)
        print("chunks:", len(doc_chunks))
        print("example chunk:", doc_chunks[:2])
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False


def main():
    init_session_state()
    set_sidebar()

    if not required_config_ready():
        st.warning(f"Please configure the required keys for {get_provider_label()} in the sidebar first.")
        return

    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # 将向量库封装成 retriever，再进一步封装为 agent 可调用的工具
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    url = st.text_input(
        ":link: Paste the blog link:",
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/",
    )
    if st.button("Enter URL"):
        if url:
            with st.spinner("Processing documents..."):
                if add_documents_to_qdrant(url, db):
                    st.success("Documents added successfully!")
                else:
                    st.error("Failed to add documents")
        else:
            st.warning("Please enter a URL")

    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?",
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query")
            return

        inputs = {"messages": [HumanMessage(content=query)]}
        with st.spinner(f"Generating response with {get_provider_label()}..."):
            try:
                response = generate_message(graph, inputs)
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    st.markdown("---")
    st.write(
        f"Current provider: :blue-background[{get_provider_label()}] | Built with :blue-background[LangChain] | :blue-background[LangGraph]"
    )


if __name__ == "__main__":
    main()
