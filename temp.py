# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import data
dataset = pd.read_csv('fraud_email_final.txt', delimiter ='\t')

#cleaning data
import re   #helps clean text efficiently
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

corpus = []  #new file of clean text
for i in range(0, dataset.shape[0]):
    text = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])  #remove unwanted text
    text = text.lower()  #lower case
    text = text.split()    # all text is splitted into list of words
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text =' '.join(text)
    corpus.append(text)


# bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  #add max_features = 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#splitting datset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Dimention Reduction - PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1350)
X_train = pca.fit_transform(X_train) #unsupervised model
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1350)
X_train = lda.fit_transform(X_train, y_train) #supervised model
X_test = lda.transform(X_test)


#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)

#logistic Regression
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)

#SVM
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, y_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier4.fit(X_train, y_train)

#predicting the test set results
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)
y_pred4 = classifier4.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)

#other metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))
print(classification_report(y_test, y_pred2))
print(classification_report(y_test, y_pred3))
print(classification_report(y_test, y_pred4))

#ROC
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred1)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)

#AUC
from sklearn.metrics import roc_auc_score
print (roc_auc_score(y_test, y_pred1))

