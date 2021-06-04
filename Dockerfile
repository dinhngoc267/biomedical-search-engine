FROM python:3.7

RUN pip install Flask gunicorn

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
ENV PORT 8080
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
#CMD [ "python3", "-m" , "flask", "run", "--host","0.0.0.0","-p", "8080"]