
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . . 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 7860

# Start backend Flask API and Streamlit UI
CMD ["sh", "-c", "python3 app.py & streamlit run ui.py --server.port 7860 --server.address 0.0.0.0"]
