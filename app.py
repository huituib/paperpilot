# -*- coding: utf-8 -*-
# 本文件是 PaperPilot（论文阅读助手）的 Streamlit 界面入口。
# 整体流程：内置示例 PDF -> 解析切块存入向量数据库 -> 选择阅读能力 -> 检索相关片段 -> 大模型生成答案 -> 展示/编辑答案和检索依据。

import hashlib
import os
from pathlib import Path
import streamlit as st

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从核心模块导入通用能力、请求构建、答案生成、PDF 索引、引用检查、公式规范化等函数。
# 真正的业务逻辑集中在 paperpilot_core.py，界面层只负责编排和展示。
from paperpilot_core import (
    CAPABILITY_OPTIONS,
    build_requests,
    citation_issues,
    generate_answer,
    index_pdf_path,
    normalize_math_markdown,
    source_label,
    stable_hash_embedding,
)


# 定义项目相关目录，全部基于当前脚本所在位置，避免写死绝对路径。
PROJECT_DIR = Path(__file__).resolve().parent          # 项目根目录（app.py 所在文件夹）
DEMO_PDF = PROJECT_DIR / "DeBEIR.pdf"                  # 内置的示例论文 PDF 路径（演示用）


# 自定义的本地哈希嵌入类：用哈希算法把文字转成向量，替代真正的语义嵌入模型。
# 继承 LangChain 的 Embeddings 接口，实现 embed_documents 和 embed_query 两个方法。
class LocalHashingEmbeddings(Embeddings):
    """Lightweight, stable vectors for the project's local RAG demo."""

    # 把一批文档片段分别转成哈希向量（用于 PDF 入库时的文本块）。
    def embed_documents(self, texts):
        return [stable_hash_embedding(text) for text in texts]

    # 把单个问题（查询）转成哈希向量。
    # 注意：必须和文档用同一个哈希函数，这样查询向量才能和文档向量做相似度比较。
    def embed_query(self, text):
        return stable_hash_embedding(text)


# 构建内置示例论文的内存索引：每个服务器进程只构建一次，之后复用。
# @st.cache_resource 保证该函数在 Streamlit 反复重跑时不被重复执行。
@st.cache_resource(show_spinner="正在为示例论文建立索引…")
def get_demo_index():
    """Build the bundled paper's in-memory index once per server process."""
    # 用 PDF 文件内容的哈希生成稳定的论文 ID，保证同一文件每次都得到相同 ID。
    paper_id = hashlib.sha256(DEMO_PDF.read_bytes()).hexdigest()[:16]
    # 创建内存向量数据库（不持久化到磁盘），集合名带上论文 ID 以避免冲突。
    db = Chroma(
        collection_name=f"paperpilot_demo_{paper_id}",
        embedding_function=LocalHashingEmbeddings(),
    )
    # 把示例 PDF 解析、切块并写入数据库，返回索引信息（页数、片段数等）。
    info = index_pdf_path(
        DEMO_PDF, DEMO_PDF.name, paper_id, db, PyPDFLoader, RecursiveCharacterTextSplitter
    )
    # 返回数据库对象和论文索引信息，供界面使用。
    return db, info


# 执行一次 RAG（检索增强生成）：根据请求检索选定论文的相关片段，并生成答案。
# 返回 (答案, 来源文档列表)。
def run_rag_chain(request, paper_id, db, chat_model):
    """Answer one paper-reading request using retrieval-augmented generation.

    Retrieve chunks from the selected paper, place their text and citations in
    the task-specific prompt, then return the answer and source Documents.
    """
    # 具体的检索、提示词组装、生成逻辑都委托给核心函数 generate_answer。
    return generate_answer(request, paper_id, db, chat_model)


# 展示答案的「检索依据」：把支撑答案的论文片段连同来源页码显示出来。
# 支持对片段文本进行会话内的临时修正（不修改数据库里的原文）。
def render_evidence(docs, paper_id, capability):
    """Show original passages and optional session-only formula corrections."""
    # 没有来源文档时不显示任何内容。
    if not docs:
        return
    # 用可折叠面板展示，默认收起，保持界面简洁。
    with st.expander("查看检索依据"):
        # 逐条显示来源片段，并为每条生成独立的编辑控件 key。
        for index, doc in enumerate(docs, start=1):
            st.markdown(f"**{index}. {source_label(doc.metadata)}**")
            excerpt = doc.page_content[:1200]            # 截取片段正文前 1200 字符展示
            st.text(excerpt)
            # 用片段内容的哈希生成唯一标识，避免不同论文/能力/索引之间控件冲突。
            digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()[:12]
            edit_base = f"paperpilot_evidence_{paper_id}_{capability}_{index}_{digest}"
            # 用户打开「修正公式/文本」开关后，展示可编辑的文本框。
            if st.toggle("修正公式 / 文本", key=f"{edit_base}_enabled"):
                editor_key = f"{edit_base}_text"
                # 首次进入时用原始片段内容初始化编辑框。
                if editor_key not in st.session_state:
                    st.session_state[editor_key] = excerpt
                # 点击「恢复抽取文本」把编辑内容还原为原始片段。
                if st.button("恢复抽取文本", key=f"{edit_base}_reset"):
                    st.session_state[editor_key] = excerpt
                edited = st.text_area("修正展示文本", key=editor_key, height=180)
                st.caption("修正预览（不改动原文或检索结果）")
                st.markdown(normalize_math_markdown(edited))   # 规范化 LaTeX/Markdown 后预览


# 编辑结果的回调函数：把文本框里的内容同步到会话状态中的答案变量。
def save_edited_answer(answer_key, widget_key):
    st.session_state[answer_key] = st.session_state[widget_key]


# 主函数：搭建整个 Streamlit 界面并管理交互流程。
def main():
    """Display the public reading demo for the bundled paper."""
    # 设置页面标题、图标和布局。
    st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
    st.title("PaperPilot")
    # 构建（或复用）示例论文的向量索引；构建失败则提示并退出。
    try:
        db, paper = get_demo_index()
    except Exception as exc:
        st.error(f"示例论文未能载入：{exc}")
        return

    # 侧边栏：用于输入 API key 和查看示例论文信息。
    with st.sidebar:
        st.header("论文工作区")
        # DeepSeek API key 输入框，保存到会话状态或环境变量。
        entered_key = st.text_input("DeepSeek API key", type="password")
        if st.button("保存 API key"):
            if entered_key.strip():
                st.session_state["deepseek_api_key"] = entered_key.strip()
                st.success("API key 已保存。")
            else:
                st.warning("请输入 API key。")

        st.divider()
        # 展示内置示例论文的标题、作者、许可链接和索引统计信息。
        st.subheader("示例论文")
        st.markdown("[DeBEIR: A Python Package for Dense Bi-Encoder Information Retrieval](https://doi.org/10.21105/joss.05017)")
        st.caption("Vincent Nguyen、Sarvnaz Karimi、Zhenchang Xing（2023） · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")
        st.caption(f"已索引 {paper.pages} 页，共 {paper.chunks} 个片段。")

    # 使用示例论文的 ID 作为当前选中的论文（本演示版只有这一篇）。
    selected_paper_id = paper.paper_id
    # 阅读能力多选框，默认勾选「论文摘要生成」。
    selected_capabilities = st.multiselect(
        "论文阅读能力（可多选）", CAPABILITY_OPTIONS,
        default=["论文摘要生成"],
    )
    # 如果勾选了「自定义问答」，则显示问题输入框。
    custom_question = ""
    if "自定义问答" in selected_capabilities:
        custom_question = st.text_area(
            "你的问题", placeholder="例如：这篇论文的样本量是多少？"
        )

    # 点击「生成所选结果」按钮后，执行 RAG 流程生成答案。
    if st.button("生成所选结果", type="primary"):
        # 从会话状态或环境变量读取 API key。
        api_key = st.session_state.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            st.warning("请先在左侧保存 DeepSeek API key。")
        else:
            # 根据所选能力构建请求列表（每个能力对应一个请求）。
            try:
                requests = build_requests(selected_capabilities, custom_question)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                # 初始化大模型（DeepSeek），所有请求共用一个模型实例。
                chat_model = ChatOpenAI(
                    model="deepseek-flash",
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                    temperature=0.2,
                )
                results = []
                # 逐个请求执行 RAG，收集答案和来源。
                for request in requests:
                    with st.spinner(f"正在生成{request.name}..."):
                        try:
                            answer, docs = run_rag_chain(
                                request, selected_paper_id, db, chat_model
                            )
                        except Exception as exc:
                            st.error(f"{request.name}生成失败：{exc}")
                        else:
                            results.append((request.name, answer, docs))
                # 为每个结果初始化「编辑用」的会话状态变量（原始答案 + 编辑框内容）。
                for name, answer, _ in results:
                    edit_key = f"paperpilot_edit_{selected_paper_id}_{name}"
                    st.session_state[edit_key] = answer
                    st.session_state[f"{edit_key}_widget"] = answer
                # 把结果存入会话状态，供刷新后继续显示。
                st.session_state["paperpilot_results"] = (selected_paper_id, results)

    # 读取会话状态中保存的结果，若结果所属论文与当前选中论文一致则重新渲染显示。
    saved = st.session_state.get("paperpilot_results")
    if saved and saved[0] == selected_paper_id:
        for name, answer, docs in saved[1]:
            edit_key = f"paperpilot_edit_{selected_paper_id}_{name}"   # 当前答案的会话状态键
            widget_key = f"{edit_key}_widget"                          # 编辑框内容的会话状态键
            # 首次显示时用原始答案初始化编辑状态。
            if edit_key not in st.session_state:
                st.session_state[edit_key] = answer
            st.subheader(name)   # 结果名称（如「论文摘要生成」）
            # 用户打开「编辑结果」开关后，展示可编辑的 Markdown/LaTeX 文本框和下载按钮。
            if st.toggle("编辑结果", key=f"paperpilot_edit_mode_{selected_paper_id}_{name}"):
                # 点击「恢复生成内容」把答案还原为大模型原始输出。
                if st.button("恢复生成内容", key=f"paperpilot_reset_{selected_paper_id}_{name}"):
                    st.session_state[edit_key] = answer
                    st.session_state[widget_key] = answer
                # 首次进入编辑模式时，用当前答案初始化编辑框。
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state[edit_key]
                # 编辑框内容变化时，通过回调同步到答案变量。
                st.text_area(
                    "编辑 Markdown / LaTeX",
                    key=widget_key,
                    height=320,
                    on_change=save_edited_answer,
                    args=(edit_key, widget_key),
                )
                rendered_answer = normalize_math_markdown(st.session_state[edit_key])   # 规范化后用于显示/下载
                # 提供 Markdown 下载按钮，文件名带论文名和结果名。
                st.download_button(
                    "下载 Markdown",
                    data=rendered_answer,
                    file_name=f"{paper.name}_{name}.md",
                    mime="text/markdown",
                    key=f"paperpilot_download_{selected_paper_id}_{name}",
                )
                st.caption("预览")
            else:
                # 非编辑模式下直接使用（可能被编辑过的）答案。
                rendered_answer = normalize_math_markdown(st.session_state[edit_key])
            st.markdown(rendered_answer)   # 渲染最终答案（支持 Markdown/LaTeX）
            # 检查答案中的引用是否与检索来源一致，不一致则给出警告。
            for issue in citation_issues(rendered_answer, docs):
                st.warning(issue)
            render_evidence(docs, selected_paper_id, name)   # 可折叠的检索依据


# 程序入口：直接运行本文件时启动 Streamlit 应用。
if __name__ == "__main__":
    main()
