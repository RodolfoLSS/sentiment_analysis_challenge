FROM python:3.8

COPY . ./sentiment_analysis_challenge
WORKDIR /sentiment_analysis_challenge

RUN pip install -r requirements.txt

EXPOSE 4000

CMD python main.py 7070