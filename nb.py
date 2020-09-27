import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import metrics
from sklearn.model_selection import KFold
#SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def runNB(dataset, n_folds = 5):
    labelEncoder = LabelEncoder()
    labelEncoder.fit(dataset['polarity'])
    target = labelEncoder.transform(dataset['polarity'])

    kf = KFold(n_splits=n_folds)

    sum_akurasi = 0

    data_list = []
    for train_index, test_index in kf.split(dataset['text']):
        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, y_train = dataset['text'].iloc[train_index], target[train_index]
        X_test, y_test = dataset['text'].iloc[test_index], target[test_index]

        print(X_train)
        print(y_train)

        stopword_file = open("stopwords-id.txt", "r+")
        stopwords = stopword_file.read()
        stopword_file.close()
        stopwords = stopwords.split("\n")

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        vectorizer = TfidfVectorizer(stop_words= stopwords, preprocessor=stemmer.stem)
        X = vectorizer.fit_transform(X_train)

        MNB = MultinomialNB()
        MNB.fit(X, y_train)

        X = vectorizer.transform(X_test)

        print("MNB Evaluation")
        Y_pred = MNB.predict(X)
        print(metrics.classification_report(y_test, Y_pred))

        akurasi = metrics.accuracy_score(y_test, Y_pred)
        print("Accuracy:",akurasi)
        sum_akurasi += akurasi
        data_list.append({
            'vectorizer': vectorizer,
            'MNB': MNB
        })
    # end for
    rata_akurasi = sum_akurasi / n_folds

    return rata_akurasi, data_list
