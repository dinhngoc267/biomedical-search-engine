FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
ENV PORT 8080
EXPOSE 8080
CMD [ "python3", "-m" , "flask", "run", "--host","0.0.0.0","-p", "8080"]