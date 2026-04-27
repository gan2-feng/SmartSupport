"""Streamlit 界面入口。"""

import streamlit as st
import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(__file__))
import agent
app = agent.app

st.set_page_config(page_title="智能客服助手", page_icon="🤖")

# 登录函数
def login():
    st.subheader("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        try:
            with open("users.json", "r", encoding="utf-8") as f:
                users = json.load(f)
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
        except FileNotFoundError:
            st.error("用户数据文件不存在")

# 检查登录状态
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    st.stop()

st.title("🤖 SmartSupport 智能客服")

def _normalize_message(msg):
    """保证消息存储为纯字符串文本。"""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if not isinstance(content, str):
        if hasattr(content, "content"):
            content = str(content.content)
        else:
            content = str(content)
    return {"role": role, "content": content}

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    st.session_state.messages = [_normalize_message(msg) for msg in st.session_state.messages]

# 为每个用户会话生成唯一的thread_id
if "thread_id" not in st.session_state:
    query_params = st.query_params
    if "thread_id" in query_params:
        st.session_state.thread_id = query_params["thread_id"]
    else:
        # 使用用户名作为 thread_id 基础
        st.session_state.thread_id = st.session_state.username
        st.info(f"会话已创建。复制此 URL 以在刷新后恢复历史: ?thread_id={st.session_state.thread_id}")

# ---- 新增：从数据库加载历史消息 ----
if "messages" not in st.session_state:
    # 尝试从 LangGraph 状态中恢复
    try:
        state = app.get_state(thread_id=st.session_state.thread_id)
        if state and "messages" in state.values:
            # 将 LangGraph 的消息对象转换为可显示的角色-内容对
            historical_messages = state.values["messages"]
            st.session_state.messages = []
            for msg in historical_messages:
                if hasattr(msg, "type"):
                    role = "user" if msg.type == "human" else "assistant"
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    st.session_state.messages.append({"role": role, "content": content})
        else:
            st.session_state.messages = []
    except Exception as e:
        st.error(f"加载历史记录失败: {e}")
        st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    user_text = str(prompt)
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 使用用户会话的唯一thread_id
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # 调用LangGraph Agent
            inputs = {"messages": [("user", user_text)]}
            result = app.invoke(inputs, config)
            # 提取最终回答
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                response = str(final_message.content)
            else:
                response = str(final_message)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})