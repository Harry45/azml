FROM python:3.9
WORKDIR /code 
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 
COPY . /code
EXPOSE  8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# this works nicely 
# docker build -t test123 .
# docker run -d --name mycontainer -p 8000:8000 test123