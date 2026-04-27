"""LangGraph Agent 核心逻辑。"""
import os
from typing import TypedDict, Annotated, Sequence, Any
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from tools import search_knowledge_base, query_order, create_ticket
from dotenv import load_dotenv
import sqlite3
import json
from contextlib import contextmanager
from langgraph.checkpoint.base import BaseCheckpointSaver, SerializerProtocol, Checkpoint, CheckpointMetadata, ChannelVersions, CheckpointTuple
from langchain_core.runnables import RunnableConfig

load_dotenv()  # 放在所有 import 之前或 os.getenv 之前

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class SqliteSaver(BaseCheckpointSaver[str]):
    """简单的 SQLite 内存保存器，用于会话隔离。"""

    def __init__(self, db_path: str = "checkpoints.db", serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    checkpoint_data TEXT,
                    metadata TEXT,
                    parent_checkpoint_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    channel TEXT,
                    value TEXT,
                    task_path TEXT,
                    write_idx INTEGER,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, channel, write_idx)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS blobs (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    channel TEXT,
                    version TEXT,
                    data TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
            ''')
            conn.commit()

            self._ensure_column(conn, 'writes', 'value_format TEXT')
            self._ensure_column(conn, 'blobs', 'data_format TEXT')
        finally:
            conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column_sql: str) -> None:
        """确保表包含指定列，如果不存在则添加。"""
        column_name = column_sql.split()[0]
        existing = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")

    def _serialize(self, value: Any) -> tuple[str, bytes]:
        return self.serde.dumps_typed(value)

    def _deserialize(self, serialized: tuple[str, bytes]) -> Any:
        return self.serde.loads_typed(serialized)

    def _deserialize_value(self, value_format: str | None, value: Any) -> Any:
        if value_format is None:
            try:
                return json.loads(value)
            except Exception:
                return value
        if not isinstance(value, bytes):
            value = value.encode()
        return self._deserialize((value_format, value))

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """获取检查点元组"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        conn = sqlite3.connect(self.db_path)
        try:
            if checkpoint_id:
                # 获取特定检查点
                row = conn.execute('''
                    SELECT checkpoint_data, metadata, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ''', (thread_id, checkpoint_ns, checkpoint_id)).fetchone()

                if row:
                    checkpoint_data, metadata, parent_checkpoint_id = row
                    checkpoint = json.loads(checkpoint_data)
                    metadata_dict = json.loads(metadata) if metadata else {}

                    # 获取写入数据
                    writes = conn.execute('''
                        SELECT task_id, channel, value_format, value
                        FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    ''', (thread_id, checkpoint_ns, checkpoint_id)).fetchall()

                    pending_writes = [
                        (
                            task_id,
                            channel,
                            self._deserialize_value(value_format, value),
                        )
                        for task_id, channel, value_format, value in writes
                    ]

                    # 加载 blobs
                    channel_values = self._load_blobs(conn, thread_id, checkpoint_ns, checkpoint["channel_versions"])

                    return CheckpointTuple(
                        config=config,
                        checkpoint={**checkpoint, "channel_values": channel_values},
                        metadata=metadata_dict,
                        pending_writes=pending_writes,
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        } if parent_checkpoint_id else None
                    )
            else:
                # 获取最新检查点
                row = conn.execute('''
                    SELECT checkpoint_id, checkpoint_data, metadata, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                ''', (thread_id, checkpoint_ns)).fetchone()

                if row:
                    checkpoint_id, checkpoint_data, metadata, parent_checkpoint_id = row
                    checkpoint = json.loads(checkpoint_data)
                    metadata_dict = json.loads(metadata) if metadata else {}

                    # 获取写入数据
                    writes = conn.execute('''
                        SELECT task_id, channel, value_format, value
                        FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    ''', (thread_id, checkpoint_ns, checkpoint_id)).fetchall()

                    pending_writes = [
                        (
                            task_id,
                            channel,
                            self._deserialize_value(value_format, value),
                        )
                        for task_id, channel, value_format, value in writes
                    ]

                    # 加载 blobs
                    channel_values = self._load_blobs(conn, thread_id, checkpoint_ns, checkpoint["channel_versions"])

                    return CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint={**checkpoint, "channel_values": channel_values},
                        metadata=metadata_dict,
                        pending_writes=pending_writes,
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        } if parent_checkpoint_id else None
                    )
        finally:
            conn.close()
        return None

    def _load_blobs(self, conn, thread_id: str, checkpoint_ns: str, versions: ChannelVersions) -> dict:
        """加载 blobs 数据"""
        channel_values = {}
        for channel, version in versions.items():
            row = conn.execute('''
                SELECT data_format, data
                FROM blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
            ''', (thread_id, checkpoint_ns, channel, str(version))).fetchone()

            if not row:
                continue

            data_format, data = row
            if data_format == "empty":
                continue

            channel_values[channel] = self._deserialize_value(data_format, data)
        return channel_values

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        """保存检查点"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        c = checkpoint.copy()
        values = c.pop("channel_values", {})

        conn = sqlite3.connect(self.db_path)
        try:
            # 保存 blobs
            for channel, version in new_versions.items():
                if channel in values:
                    data_format, data_payload = self._serialize(values[channel])
                else:
                    data_format, data_payload = "empty", b""
                conn.execute('''
                    INSERT OR REPLACE INTO blobs (thread_id, checkpoint_ns, channel, version, data, data_format)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    thread_id,
                    checkpoint_ns,
                    channel,
                    str(version),
                    sqlite3.Binary(data_payload),
                    data_format,
                ))

            # 保存检查点
            parent_checkpoint_id = config["configurable"].get("checkpoint_id")
            conn.execute('''
                INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint_data, metadata, parent_checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (thread_id, checkpoint_ns, checkpoint_id, json.dumps(c), json.dumps(metadata), parent_checkpoint_id))

            conn.commit()
        finally:
            conn.close()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """保存写入数据"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        conn = sqlite3.connect(self.db_path)
        try:
            for idx, (channel, value) in enumerate(writes):
                fmt, payload = self._serialize(value)
                conn.execute('''
                    INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, channel, value, task_path, write_idx, value_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    channel,
                    sqlite3.Binary(payload),
                    task_path,
                    idx,
                    fmt,
                ))

            conn.commit()
        finally:
            conn.close()

    def delete_thread(self, thread_id: str) -> None:
        """删除线程的所有数据"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('DELETE FROM checkpoints WHERE thread_id = ?', (thread_id,))
            conn.execute('DELETE FROM writes WHERE thread_id = ?', (thread_id,))
            conn.execute('DELETE FROM blobs WHERE thread_id = ?', (thread_id,))
            conn.commit()
        finally:
            conn.close()

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

memory = SqliteSaver()
app = workflow.compile(checkpointer=memory)