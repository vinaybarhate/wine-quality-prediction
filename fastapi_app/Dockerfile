# Ue official directory inside container 
FROM python:3.11-slim

# set working 
WORKDIR /app

# Copy requirements
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy entire project including model & templates
COPY . .

EXPOSE 8000

CMD ["uvicorn", "Fast:app", "--host", "0.0.0.0", "--port", "8000"]
