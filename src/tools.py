"""工具定义：RAG 检索、订单查询、创建工单等。"""

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 初始化向量库（供RAG工具使用）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@tool
def search_knowledge_base(query: str) -> str:
    """查询公司内部知识库，获取产品政策、退换货规则、物流信息等。"""
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return "未找到相关信息。"
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def query_order(phone: str) -> str:
    """根据用户手机号查询最近一笔订单的状态。"""
    import json
    try:
        with open("orders.json", "r", encoding="utf-8") as f:
            orders = json.load(f)
        return orders.get(phone, "未查询到该手机号对应的订单，请确认号码是否正确。")
    except FileNotFoundError:
        return "订单数据文件不存在。"

@tool
def create_ticket(user_issue: str, contact: str) -> str:
    """当问题无法立即解决时，创建一张人工客服工单。"""
    ticket_id = f"TK{hash(user_issue) % 10000:04d}"
    # 写入文件或数据库
    with open("tickets.txt", "a", encoding="utf-8") as f:
        f.write(f"{ticket_id},{contact},{user_issue}\n")
    return f"工单已创建，工单号：{ticket_id}，客服将在1个工作日内联系您。"