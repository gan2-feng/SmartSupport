"""LangGraph Agent 核心逻辑。"""
import os
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from tools import search_knowledge_base, query_order, create_ticket
from dotenv import load_dotenv

load_dotenv()  # 放在所有 import 之前或 os.getenv 之前

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 定义Agent状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[tuple], operator.add]  # 对话历史
    user_context: dict  # 可存储用户手机号等上下文

# 工具列表
tools = [search_knowledge_base, query_order, create_ticket]


# 初始化LLM
llm = ChatOpenAI(model="deepseek-chat", 
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com",
                temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

# 定义节点函数
def agent_node(state: AgentState):
    """Agent决策节点：决定调用工具还是直接回答"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    # 直接调用 ToolNode 实例，它内部会自动处理工具的执行和结果返回
    return ToolNode(tools).invoke(state)

def should_continue(state: AgentState):
    """判断是否需要继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tool_node": "tool_node", END: END})
workflow.add_edge("tool_node", "agent")

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)