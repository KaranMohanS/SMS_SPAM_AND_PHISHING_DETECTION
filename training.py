# for numerical operations
import pandas as pd
import numpy as np

# for graphical visualization
import matplotlib.pyplot as plt
import seaborn as sns

import string

import re              #importing necessary NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

df=pd.read_csv('/home/karan/Downloads/data/spam.csv' , encoding = "ISO-8859-1")


print(df.head())

print(df.columns)

print(df.shape)


df=df.drop('Unnamed: 2',axis=1)
df=df.drop('Unnamed: 3',axis=1)
df=df.drop('Unnamed: 4',axis=1)



df=df.rename(columns={'v1':'target','v2':'text'})
print(df.head())

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

df = df.drop_duplicates(keep='first')

df['num_characters'] = df['text'].apply(len)

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


df['tranformed_text'] = df['text'].apply(transform_text)

spam_corpus = []
for msg in df[df['target']==1]['tranformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)     

print(len(spam_corpus))

ham_corpus = []
for msg in df[df['target']==0]['tranformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word) 

print(len(ham_corpus))


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['tranformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred1 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

query = "WIN: We have a winner! Mr. T. Foley won an iPod! More exciting prizes soon, so keep an eye on ur mobile or visit www.win-82050.co.uk"
q2 = "REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls reply 2 this text with your valid name, house no and postcode"
q3 = "WIN: We have a winner! Mr. T. Foley won an iPod! More exciting prizes soon, so keep an eye on ur mobile or visit www.win-82050.co.uk"
x_q = tfidf.transform([query,q2,q3]).toarray()

res = mnb.predict(x_q)

print(res)


# lemmatizer=WordNetLemmatizer()              #object creation for lemmatzation on corpus of data
# corpus=[]


# for i in range(0,len(df)):
#     review=re.sub('[^a-zA-Z]','',df['message'][i])
#     review=review.lower()
#     review=review.split()
#     review=[lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
#     review=' '.join(review)
#     corpus.append(review)

# df.head()

# cv=CountVectorizer(max_features=1000) 

# X=cv.fit_transform(corpus).toarray()  

# y=pd.get_dummies(df['label']) 

# y=y.iloc[:,1].values

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# spam_detect_model=MultinomialNB().fit(X_train,y_train)

# y_pred=spam_detect_model.predict(X_test)

# print(y_pred)

# print(accuracy_score(y_test,y_pred))

# query = "Free free free free Free free free free Free free free free Free free free free Free free free free Free free free free Free free free free Free free free free"

# x_q = cv.transform([query])

# res = spam_detect_model.predict(x_q)

# print(res)