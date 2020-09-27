#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install sastrawi')


# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import metrics

#SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd


# %%

# In[4]:


df_train=pd.read_csv('hasilCrawling.csv')


# In[5]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df_train['category'])
target = labelEncoder.transform(df_train['category'])


# In[6]:


stopword_file = open("stopwords-id.txt", "r+")
stopwords = stopword_file.read()
stopword_file.close()
stopwords = stopwords.split("\n")
#stopwords


# In[7]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[8]:


#stopword=open('https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt',"r+")
vectorizer = TfidfVectorizer(stop_words= stopwords, preprocessor=stemmer.stem)
#vectorizer = TfidfVectorizer(stop_words= stopwords)
X = vectorizer.fit_transform(df_train['text'])
#X


# In[9]:


X = vectorizer.transform(df_train['text'])


# In[10]:


#vectorizer.get_feature_names()


# In[11]:


#svd = TruncatedSVD(n_components=3)
#vector_reduced = svd.fit_transform(X)


# In[12]:


classifier = SVC(kernel="rbf")
classifier.fit(X, target)


# In[13]:


MNB = MultinomialNB()
MNB.fit(X, target)


# In[14]:


from joblib import dump
dump(vectorizer, "tfidfvectorizer-prod.joblib")
dump(classifier, "svm-rbf-prod.joblib")
dump(MNB, "mnb-prod.joblib")
dump(labelEncoder, "labelEncoder.joblib")


# In[15]:


df_test = pd.read_csv("hasilCrawling.csv")
X = vectorizer.transform(df_test['text'])
Y_actual = labelEncoder.transform(df_test['category'])


# In[15]:


print("SVM RBF Evaluation")
Y_pred = classifier.predict(X)
print(metrics.classification_report(Y_actual, Y_pred))


# In[16]:


print("MNB Evaluation")
Y_pred = MNB.predict(X)
print(metrics.classification_report(Y_actual, Y_pred))


# In[17]:


from sklearn import metrics


# In[18]:


print("Accuracy:",metrics.accuracy_score(Y_actual, Y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




