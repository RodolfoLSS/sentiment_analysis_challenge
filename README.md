# Sentiment Analysis Challenge

This repo implements a sentiment analysis challenge.

- `data/`: Contains a .csv file to be used in the model training and a .txt with a couple of examples to test the API endpoint `/predict`.
- `model/`: Contains .pkl file with the sentiment analysis classification model.
- `modules/`: Implements modules used in the API endpoints.
- `Dockerfile`: Instructions to build Docker image.
- `challenge.ipynb`: Notebook applying exploratory data analysis and benchmarking models.
- `main.py`: Python file implementing four API endpoints.
## How to use

#### Running locally

Install the requirements.

```bash
pip install -r requirements.txt
```

Choose a port and execute the script.

```bash
python main.py [choose_a_port]
```

Your API is accessible at localhost:[choose_a_port].

#### Run in a Docker container

First of all, you need to build the image.

```bash
docker build -t sentiment_analysis_challenge -f Dockerfile .
```
Then, you have to run the image. 

```bash
docker run -d -p 4000:7070 sentiment_analysis_challenge
```

Your API is accessible at localhost:4000.
## Endpoints

The API contains four endpoints:

- `train/`: Trains the sentiment analysis classification model.
- `predict/`: Predicts the sentiment of comments.
- `load/`: Load sentiment analysis classification model from pickle file.
- `wipe/`: Remove sentiment analysis classification model file.