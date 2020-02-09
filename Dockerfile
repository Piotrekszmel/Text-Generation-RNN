FROM python:3.6

EXPOSE 5000
WORKDIR /TextGen
COPY requirements.txt /TextGen
RUN pip install -r requirements.txt
COPY . /TextGen
CMD ["python", "app.py"]
