"""Streamlit 界面入口。"""

import streamlit as st
from agent import app

st.set_page_config(page_title="智能客服助手", page_icon="🤖")
st.title("🤖 SmartSupport 智能客服")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 为每个会话生成一个唯一的thread_id
            config = {"configurable": {"thread_id": "user_123"}}

            # 调用LangGraph Agent
            inputs = {"messages": [("user", prompt)]}
            result = app.invoke(inputs, config)
            # 提取最终回答
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                response = final_message.content
            else:
                response = str(final_message)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})