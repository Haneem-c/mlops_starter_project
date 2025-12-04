# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all files from your project into the container
COPY . .

# 4. Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Default command: run your training script
CMD ["python", "train_model.py"]
