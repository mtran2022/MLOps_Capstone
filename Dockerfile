FROM python:3.8.1-slim
EXPOSE 8080
WORKDIR /app
COPY . /app
RUN pip install --no-cache -r requirements.txt
ENV PYTHONPATH /app
CMD ["uvicorn","--host","0.0.0.0","--port","8080","app_2mod.CC_fraud_2mod:app"]