import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from modules.utils import stopwords

stemmer = SnowballStemmer("english")


def normalize_data(data: pd.DataFrame):
    data.comment = data.comment.str.lower()
    data.drop_duplicates(["user_id", "game", "sentiment", "comment"], inplace=True)


def remove_stop_words(sentence):
    re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)
    return re_stop_words.sub("", sentence)


def remove_numbers(sentence):
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    return sentence


def decontract(token):
    token = re.sub(r"won't", "will not", token)
    token = re.sub(r"can\'t", "can not", token)
    token = re.sub(r"n\'t", " not", token)
    token = re.sub(r"\'re", " are", token)
    token = re.sub(r"\'s", " is", token)
    token = re.sub(r"\'d", " would", token)
    token = re.sub(r"\'ll", " will", token)
    token = re.sub(r"\'t", " not", token)
    token = re.sub(r"\'ve", " have", token)
    token = re.sub(r"\'m", " am", token)
    return token


def remove_punctuation(token):
    token = re.sub(r'[?|!|\'|"|#]', r"", token)
    token = re.sub(r"[.|,|)|(|\|/]", r" ", token)
    token = token.strip()
    token = token.replace("\n", " ")
    return token


def stem(token):
    stem_token = ""
    for word in token.split():
        stem = stemmer.stem(word)
        stem_token += stem
        stem_token += " "
    stem_token = stem_token.strip()
    return stem_token


def decode_sentiment(df):
    df["positive"] = 0
    df["negative"] = 0
    df["neutral"] = 0
    df["irrelevant"] = 0
    df.loc[df["sentiment"].str.lower() == "positive", ["positive"]] = 1
    df.loc[df["sentiment"].str.lower() == "negative", ["negative"]] = 1
    df.loc[df["sentiment"].str.lower() == "neutral", ["neutral"]] = 1
    df.loc[df["sentiment"].str.lower() == "irrelevant", ["irrelevant"]] = 1

    return df


def preprocess_comment(df: pd.DataFrame):
    df.comment = df.comment.apply(lambda x: decontract(str(x)))
    df.comment = df.comment.apply(remove_stop_words)
    df.comment = df.comment.apply(remove_punctuation)
    df.comment = df.comment.apply(remove_numbers)
    df["comment"] = df.comment.apply(stem)
    return df
