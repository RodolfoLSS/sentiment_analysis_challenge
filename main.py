import sys
import os
import shutil
import time
import traceback
import json
import pickle

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocessing import preprocess_comment
from modules.utils import encode_sentiment, decode_sentiment

app = Flask(__name__)

# inputs
training_data = "data/labelled_text.csv"

model_directory = "model"
model_file_name = "%s/model.pkl" % model_directory

# Ppulated at training time
clf = None
vectorizer = None


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
    df = encode_sentiment(df)
    df_preprocessed = preprocess_comment(df)
    X_train = df_preprocessed["comment"]
    y_train = df_preprocessed[df_preprocessed.columns[4:]]

    global vectorizer
    vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2")
    X_train = vectorizer.fit_transform(X_train)

    global clf
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced"), n_jobs=-1)
    start = time.time()
    clf.fit(X_train, y_train)

    pickle.dump(clf, open(model_file_name, "wb"))

    message1 = "Trained in %.5f seconds" % (time.time() - start)
    message2 = "Model training score: %s" % clf.score(X_train, y_train)
    return_message = "Success. \n{0}. \n{1}.".format(message1, message2)
    return return_message


@app.route("/load", methods=["GET"])
def load():
    try:
        global clf
        clf = pickle.load(open(model_file_name, "rb"))
        return_message = "Success loading model."
    except Exception as e:
        print(str(e))
        return_message = "No model available. Train first!"

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

    app.run(host="0.0.0.0", port=port, debug=True)
