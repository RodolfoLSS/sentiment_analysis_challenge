import sys
import os
import shutil
import time
import traceback
import json
import pickle

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocessing import preprocess_comment, decode_sentiment

app = Flask(__name__)

# inputs
training_data = "data/labelled_text.csv"
include = ["Age", "Sex", "Embarked", "Survived"]
dependent_variable = include[-1]

model_directory = "model"
model_file_name = "%s/model.pkl" % model_directory

vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2")

# These will be populated at training time
clf = None


@app.route("/predict", methods=["POST"])
def predict():
    if clf:
        try:
            json_ = request.json
            input_data_df = pd.DataFrame.from_dict(json_)
            df_preprocessed = preprocess_comment(input_data_df)

            X_test = df_preprocessed["comment"]
            global vectorizer
            X_test = vectorizer.transform(X_test)

            predictions = clf.predict(X_test)

            def decode_sentiment(sentiments: list):
                predictions_decoded = []
                for sentiment in sentiments:
                    if sentiment[0]:
                        predictions_decoded.append("positive")
                    elif sentiment[1]:
                        predictions_decoded.append("negative")
                    elif sentiment[2]:
                        predictions_decoded.append("neutral")
                    else:
                        predictions_decoded.append("irrelevant")
                return predictions_decoded

            predictions_decoded = decode_sentiment(predictions)

            return json.dumps(predictions_decoded)

        except Exception as e:

            return jsonify({"error": str(e), "trace": traceback.format_exc()})
    else:
        print("train first")
        return "no model here"


@app.route("/train", methods=["GET"])
def train():
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(training_data, header=None, encoding="ISO-8859-1")
    df.columns = ["user_id", "game", "sentiment", "comment"]
    df = decode_sentiment(df)
    df_preprocessed = preprocess_comment(df)
    X_train = df_preprocessed["comment"]
    y_train = df_preprocessed[df_preprocessed.columns[4:]]

    global vectorizer
    X_train = vectorizer.fit_transform(X_train)

    global clf
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced"), n_jobs=-1)
    start = time.time()
    clf.fit(X_train, y_train)

    pickle.dump(clf, open(model_file_name, "wb"))
    # joblib.dump(clf, model_file_name)

    message1 = "Trained in %.5f seconds" % (time.time() - start)
    message2 = "Model training score: %s" % clf.score(X_train, y_train)
    return_message = "Success. \n{0}. \n{1}.".format(message1, message2)
    return return_message


@app.route("/load", methods=["GET"])
def load():
    global clf
    clf = pickle.load(open(model_file_name, "rb"))
    return_message = "Success loading model."
    return return_message


@app.route("/wipe", methods=["GET"])
def wipe():
    try:
        shutil.rmtree("model")
        os.makedirs(model_directory)
        return "Model wiped"

    except Exception as e:
        print(str(e))
        return "Could not remove and recreate the model directory"


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except Exception:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print("model loaded")

    except Exception as e:
        print("No model here")
        print("Train first")
        print(str(e))
        clf = None

    app.run(host="0.0.0.0", port=port, debug=True)
