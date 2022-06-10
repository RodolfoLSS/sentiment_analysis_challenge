from sklearn.linear_model import LogisticRegression

# from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from preprocessing import decontract, remove_numbers, remove_punctuation, remove_stop_words, stem

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.info("Ingest data")
    filepath = "/Users/rodolfo.saldanha/Desktop/sentiment_analysis_challenge/labelled_text.csv"
    data = pd.read_csv(filepath, header=None, encoding="ISO-8859-1")
    logging.info(f"The data has {len(data)} rows")
    data.columns = ["user_id", "game", "sentiment", "comment"]
    data["positive"] = 0
    data["negative"] = 0
    data["neutral"] = 0
    data["irrelevant"] = 0
    data.loc[data["sentiment"].str.lower() == "positive", ["positive"]] = 1
    data.loc[data["sentiment"].str.lower() == "negative", ["negative"]] = 1
    data.loc[data["sentiment"].str.lower() == "neutral", ["neutral"]] = 1
    data.loc[data["sentiment"].str.lower() == "irrelevant", ["irrelevant"]] = 1

    seed = 1

    logging.info("Preprocess comments")
    data.comment = data.comment.apply(lambda x: decontract(str(x)))
    data.comment = data.comment.apply(remove_stop_words)
    data.comment = data.comment.apply(remove_punctuation)
    data.comment = data.comment.apply(remove_numbers)
    data["comment"] = data.comment.apply(stem)
    logging.info("Split train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        data["comment"], data[data.columns[4:]], test_size=0.3, random_state=seed, shuffle=True
    )
    vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2")

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    # print("AUC ROC for MultiOutputClassifier is {}".format(roc_auc_score(y_test, predictions)))

    # classifier = BinaryRelevance(GaussianNB()).fit(X_train, y_train)
    # predictions = classifier.predict(X_test)
    # print("AUC ROC for BinaryRelevance is {}".format(roc_auc_score(y_test, predictions.toarray())))

    # classifier = ClassifierChain(LogisticRegression()).fit(X_train, y_train)
    # predictions = classifier.predict(X_test)
    # print("AUC ROC for ClassifierChain is {}".format(roc_auc_score(y_test, predictions.toarray())))

    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced"), n_jobs=-1).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("AUC ROC for ClassifierChain is {}".format(roc_auc_score(y_test, predictions)))
    joblib.dump(clf, "model.pkl")
