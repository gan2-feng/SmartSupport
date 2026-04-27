FROM python:3.10-slim
WORKDIR /app
# 安装依赖
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir \
    --timeout 300 \
    --retries 5 \
    -i https://mirrors.cloud.tencent.com/pypi/simple/ \
    -r requirements.txt
# 复制项目文件
COPY src/ src/
COPY knowledge_base/ knowledge_base/
# 暴露Streamlit默认端口
EXPOSE 8501
# 启动命令
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]