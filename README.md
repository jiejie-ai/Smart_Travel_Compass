# Smart_Travel_Compass
一个从 0 到 1 的 **RAG + Agent** 落地项目。用户输入一个目的地，智能体自动完成
  「查询改写 → 联网搜索 → 知识库检索 → 天气查询 → 图片搜索 → 生成 PDF 攻略 → 上传云端」的全流程。

  ## ✨ 核心能力

  - **Agent 编排**：LangGraph 状态机（check_state → thinker → actor → finalize），串联 10+ 工具
  - **RAG 知识库**：RAGFlow 向量检索 + 查询改写 + 多查询扩展，提升召回率
  - **多数据源融合**：Tavily 联网搜索 / RAGFlow 本地知识库 / 高德地图 / Pexels 图片
  - **MCP 协议扩展**：自研 FastMCP 图片搜索 Server + 第三方高德 MCP
  - **流式体验**：SSE 实时回传 LLM token 与工具执行进度
  - **PDF 生成**：ReportLab 图文排版 + 腾讯云 COS 上传，提供下载链接

  ## 🧱 技术栈

  Python · FastAPI · LangChain/LangGraph · DeepSeek · RAGFlow · Tavily · 高德地图 ·
  Pexels · MCP · ReportLab · 腾讯云 COS

  ## 🚀 快速开始

  ```bash
  .venv\Scripts\activate
  uvicorn src.main:app --reload --host 0.0.0.0 --port 8123

  访问 http://localhost:8123
