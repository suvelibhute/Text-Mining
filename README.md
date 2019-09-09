# Phishing Email Detection using various Machine Learning Techniques
## Problem Statement
Email Communication is one of the most common modes of communication. However,
phishing email is one of the most serious Internet phenomenon. Phishing is a form of cybercrime where an attacker imitates a real person/institution by promoting them as an official person or entity through e-mail or other communication mediums. Hence, we need to find out different techniques to detect fraudulent emails.

## Objective
Building different ML techniques to identify phishing email and comparing the performance of these ML techniques with respect to different metrics.

## Dataset
fraud_email_final.tsv is a dataset from Kaggle.com. It consists 2 columns.  The first column consists of 11881 text emails. The second column tells us whether an email is legitimate(0) or phishing(1).
Software
Spyder with python 3.7
## Procedure
For building a machine learning to predict whether an email is legitimate or phishing, following steps are performed.
1.	Cleaning Text data:
I.	Imported the dataset(tsv file) into Spyder(Python 3.7) using Pandas library
II.	Text cleaning and pre-processing:
•	Removed Punctuation and numbers from text by selecting only text format using re library
•	Converted text to lower case
•	The String of text is converted into list of words
•	Removed stop words using nltk library
•	Lemmatization – It is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
•	Again joined the list of words with a space in between using function “join”
•	Created a bag of words model. It is nothing but we use the tokenized words for each observation and find out the frequency of each token. Generating a sparse matrix with max 1500 features


2.	Dimensionality Reduction:
Each observation is represented by thousands of tokens, which make the classification problem very hard for many classifiers. Dimensionality reduction is a typical step in text mining, which makes the dataset short and compact. The new space is easier to handle because of its size, and also to carry the most important part of the information needed to distinguish between emails, allowing for the creation of profiles that describe the data set. Two major classes of dimensionality reduction techniques are feature extraction and feature selection. I have used feature extraction techniques – PCA and LDA in this project
3.	Prediction model: Built various machine learning models - 
    I. I have built random forest, logistic regression, SVM and Naive Bayes model with number of features = 1500
    II. I have also built a Naive Bayes model with PCA (Number of features = 900, 1000, 1200, 1300, 1350)
    III. I have built a Naive Bayes model with LDA (Number of features = 1000, 900, 1200, 1300, 1350)

## Observation
1.	Area Under ROC for various ML algorithms without dimensionality reduction(Number of features = 1500):
    RF = 0.9911,
    Logistic Regression = 0.9855,
    Naive Bayes = 0.9715,
    SVM = 0.9896
                      
2.	Area Under ROC for a ML algorithm with dimesionality reduction methods - PCA and LDA:
    Naive Bayes with PCA(Number of features = 900) = 0.9515,
    Naive Bayes with PCA(Number of features = 1000) = 0.8454,
    Naive Bayes with PCA(Number of features = 1200) = 0.8511,
    Naive Bayes with PCA(Number of features = 1300) = 0.8519,
    Naive Bayes with PCA(Number of features = 1350) = 0.8518,
    
    Naive Bayes with LDA(Number of features = 900) = 0.9515,
    Naive Bayes with PCA(Number of features = 1000) = 0.9575,
    Naive Bayes with PCA(Number of features = 1200) = 0.9488,
    Naive Bayes with PCA(Number of features = 1300) = 0.9588,
    Naive Bayes with PCA(Number of features = 1350) = 0.9587 
                         

## Conclusion
•	Random Forest model outperforms other methods in terms of the area under the ROC curve(0.9911) 
•	Also, after applying feature extraction techniques like PCA and LDA, we are getting similar results(Area under ROC = 0.9511) with less number of features (900)
