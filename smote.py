import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
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

df = pd.read_csv('hasilCrawling terbaru.csv')

X = df[['no', 'text']]
Y = df[['polarity']]
print(Y)
print(df.groupby('polarity').size())
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
sm = SMOTE(random_state=1)
X1, Y1_b = sm.fit_resample(X, Y)
print(Y1_b)
print(Y1_b.groupby('polarity').size())
new_df = pd.concat([pd.DataFrame(X1), pd.DataFrame(Y1_b)], axis=1)
print(new_df)
new_df.to_csv('hasilCrawling SMOTE.csv')