FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "python3 app.py & streamlit run ui.py --server.port=7860 --server.address=0.0.0.0"]
