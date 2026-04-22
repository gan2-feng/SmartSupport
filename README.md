# 智能客服项目

项目结构：

├── knowledge_base/                 # 原始知识文档
│   ├── 退货政策.txt
│   ├── 物流说明.txt
│   └── 产品FAQ.txt
├── chroma_db/                      # 向量库持久化目录（自动生成）
├── src/
│   ├── build_vectorstore.py        # 1. 构建向量库脚本
│   ├── tools.py                    # 2. 工具定义（RAG检索、订单查询、创建工单）
│   ├── agent.py                    # 3. LangGraph Agent核心逻辑
│   └── app.py                      # 4. Streamlit界面入口
├── requirements.txt
└── README.md

## 说明

- `knowledge_base/` 包含示例知识文档，可用于构建向量检索数据库。
- `chroma_db/` 为向量库持久化目标目录，运行时会自动生成内容。
- `src/` 下为项目核心代码。

## 快速开始

1. 创建并激活 Python 环境
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行 Streamlit 应用：
   ```bash
   streamlit run src/app.py
   ```
