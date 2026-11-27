# PaperPilot

PaperPilot 是面向研究生的论文阅读 RAG 示例。打开页面后，可以多选论文摘要、方法解释、组会笔记和自定义问答。每项结果下方都可以查看检索到的原文片段及页码。

本项目改造自 `awesome-llm-apps` 的 PharmaQuery 示例，保留了原始项目的 PDF、Chroma 和 Streamlit 主流程。

## 本项目的处理流程

```text
内置 DeBEIR.pdf -> 按页提取文本 -> 1000 字符切块（重叠 150）
-> 本地稳定哈希向量 -> 内存 Chroma 索引
-> 选择阅读能力 -> 检索该论文相关片段
-> DeepSeek 依据片段生成结果 -> 显示回答与来源页码
```

演示版随项目分发原版 [DeBEIR.pdf](DeBEIR.pdf)。Vincent Nguyen、Sarvnaz Karimi 和 Zhenchang Xing 于 2023 年在 JOSS 发表了这篇论文。[论文来源](https://doi.org/10.21105/joss.05017) · [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)。

服务首次打开页面时从本地 PDF 建立索引。同一服务进程中的刷新和新会话复用索引，服务重启后自动重建。部署时将 PDF 与 `app.py` 放在同一目录。

## 本地运行（Windows PowerShell）

在 VS Code 中打开本目录作为工作区，然后在终端运行：

```powershell
cd D:\codex\学习\awesome-llm-apps-main\rag_tutorials\rag_chain
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8503 --browser.gatherUsageStats false
```

浏览器打开 `http://localhost:8503`。也可以在 VS Code 中选择运行配置 `PaperPilot (Streamlit)`；直接执行 `python app.py` 不会启动网页服务。

建议使用 Python 3.12；在其他电脑上把第一行路径换成该电脑存放项目的位置即可。依赖只安装到当前项目的 `.venv`。

侧边栏输入并保存 DeepSeek API key。勾选一个或多个阅读能力，再点击“生成所选结果”。也可在启动前设置 `DEEPSEEK_API_KEY` 环境变量；不要将密钥写入仓库。

每项结果可打开“编辑结果”，修改 Markdown/LaTeX 后在下方预览，或下载为 Markdown 文件。行内公式使用 `$...$`，独立公式使用单独成行的 `$$...$$`；编辑内容仅保存在当前浏览器会话，重启后需从下载文件恢复。答案若缺少页码引用，或引用了本次未检索到的页码标签，页面会给出警告；零检索结果时直接拒答。

“查看检索依据”中每个片段保留只读的 PDF 抽取原文。展开“修正公式 / 文本”可单独编辑展示稿、预览 LaTeX 公式，或恢复抽取文本。修正稿只在当前会话中显示，不会写回 Chroma，也不会改变已生成答案使用的证据。

## 当前限制

本地哈希向量是轻量级演示方案，语义检索质量弱于专门训练的 Embedding 模型。中文自定义问题会通过 DeepSeek 增补英文检索关键词，因此会多一次 API 调用。固定任务只检索相关片段，不保证覆盖整篇长论文；引用标签属于本次检索结果，不代表原文真的支持该结论，重要结论仍需点击“查看检索依据”核对原文。页码指 PDF 页面顺序，可能与论文印刷页码不同。详细设计见 [引用与防幻觉设计说明](../../../PaperPilot引用与防幻觉设计说明.md)。
