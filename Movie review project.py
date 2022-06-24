#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd 
import re
import string
from nltk.stem import WordNetLemmatizer  
import seaborn as sns 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud 
import nltk
nltk.download('stopwords')
import re 
from nltk.sentiment.vader import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from nltk.stem import WordNetLemmatizer   

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pip install wordcloud


# In[3]:


pip install vaderSentiment


# In[4]:


# Reading the dataset
dataset = pd.read_csv('movies.tsv', delimiter = '\t', quoting = 3)


# In[5]:


dataset.head()


# In[6]:


#dividing positive and negative reviews
print("The number of Positive movies=", (dataset["Liked"]==1).sum())
print("The number of Negative movies=", (dataset["Liked"]==0).sum())


# In[7]:


pos_mov=" "
neg_mov=" "


# In[8]:


len(dataset)


# In[9]:


for i in range(0,99):
    if dataset["Liked"][i]==1:
        pos_mov=pos_mov+" "+dataset["Movie"][i]
    if dataset["Liked"][i]==0:
        neg_mov=neg_mov+" "+dataset["Movie"][i]


# In[10]:


print("Length of positive reviews-",len(pos_mov))
print("")
print("Part of positive reviews-")
print(pos_mov[1000:2000])


# In[11]:


print("Length of negative reviews-",len(neg_mov))
print("")
print("Part of negative reviews-")
print(neg_mov[1000:2000])


# In[12]:


#positive reviews
import string
string.punctuation


# In[13]:


#removing the punctuations

text_nopunct_pos=''

text_nopunct_pos= "".join([char for char in pos_mov if char not in string.punctuation])


# In[14]:


#Creating the tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')


# In[15]:


#Tokenizing the text
pos_tokens = tokenizer.tokenize(text_nopunct_pos)
len(pos_tokens)


# In[16]:


#now we shall make everything lowercase for uniformity
#to hold the new lower case words

words_pos = []

# Looping through the tokens and make them lower case
for word in pos_tokens:
    words_pos.append(word.lower())


# In[17]:


#Stop words are generally the most common words in a language.
#English stop words from nltk.

stopwords = nltk.corpus.stopwords.words('english')


# In[18]:


final_words_pos=[]

#Now we need to remove the stop words from the words variable
#Appending to words_new all words that are in words but not in sw

for word in words_pos:
    if word not in stopwords:
        final_words_pos.append(word)


# In[19]:


wn = WordNetLemmatizer()


# In[20]:


lem_words_pos=[]

for word in final_words_pos:
    word=wn.lemmatize(word)
    lem_words_pos.append(word)


# In[21]:


import nltk
nltk.download('wordnet')


# In[22]:


#The frequency distribution of the words
freq_dist_pos = nltk.FreqDist(lem_words_pos)


# In[23]:


#Frequency Distribution Plot
plt.subplots(figsize=(20,12))
freq_dist_pos.plot(30)


# In[24]:


#converting into string

res_pos=' '.join([i for i in lem_words_pos if not i.isdigit()])


# In[68]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=100,
                          width=1400,
                          height=1200
                         ).generate(res_pos)


plt.imshow(wordcloud)
plt.title('Positive Reviews Word Cloud (100 words)')
plt.axis('off')
plt.show()


# In[69]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=200,
                          width=1400,
                          height=1200
                         ).generate(res_pos)


plt.imshow(wordcloud)
plt.title('Positive Reviews Word Cloud (200 words)')
plt.axis('off')
plt.show()


# In[27]:


text_nopunct_neg=''

text_nopunct_neg= "".join([char for char in neg_mov if char not in string.punctuation])


# In[28]:


#Tokenizing the text
neg_tokens = tokenizer.tokenize(text_nopunct_neg)
len(neg_tokens)


# In[29]:


#now we shall make everything lowercase for uniformity
#to hold the new lower case words

words_neg = []

# Looping through the tokens and make them lower case
for word in neg_tokens:
    words_neg.append(word.lower())


# In[30]:


final_words_neg=[]

#Now we need to remove the stop words from the words variable
#Appending to words_new all words that are in words but not in sw

for word in words_neg:
    if word not in stopwords:
        final_words_neg.append(word)


# In[31]:


lem_words_neg=[]

for word in final_words_neg:
    word=wn.lemmatize(word)
    lem_words_neg.append(word)


# In[32]:


#The frequency distribution of the words
freq_dist_neg = nltk.FreqDist(lem_words_neg)


# In[33]:


#Frequency Distribution Plot
plt.subplots(figsize=(20,12))
freq_dist_neg.plot(30)


# In[34]:


#converting into string

res_neg=' '.join([i for i in lem_words_neg if not i.isdigit()])


# In[70]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=100,
                          width=1400,
                          height=1200
                         ).generate(res_neg)


plt.imshow(wordcloud)
plt.title('Negative Reviews Word Cloud (100 words)')
plt.axis('off')
plt.show()


# In[71]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=200,
                          width=1400,
                          height=1200
                         ).generate(res_neg)


plt.imshow(wordcloud)
plt.title('Negative Reviews Word Cloud (200 words)')
plt.axis('off')
plt.show()


# In[37]:


#Review classifier
import nltk
import pandas as pd
import re
import string
from nltk.stem import WordNetLemmatizer


# In[38]:


stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# In[39]:


# Preprocessing
nltk.download('stopwords')
corpus = []
for i in range(0, 99):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Movie'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# In[40]:


#count vectorization
from sklearn.feature_extraction.text import CountVectorizer


# In[41]:


# Creating the Bag of Words model
cv = CountVectorizer(max_features = 99)

#the X and y
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[42]:


import numpy as np
from sklearn.model_selection import train_test_split
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 3)
y_test= np.nan_to_num(y_test)


# In[43]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[54]:


#using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
# Random Forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[55]:


from sklearn.metrics import accuracy_score,confusion_matrix
print("the accuracy level is=",accuracy_score(y_pred,y_test)*100,"%")
cm = confusion_matrix(y_pred,y_test)
print(cm)


# In[56]:


#accuracy score and classification report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[57]:


accuracy_score(y_test, y_pred)


# In[58]:


print(classification_report(y_test, y_pred))


# In[60]:


#using SVM

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)


# In[61]:


y_pred=classifier.predict(X_test)


# In[62]:


print(y_pred,y_test)


# In[63]:


from sklearn.metrics import accuracy_score,confusion_matrix
print("the accuracy level is=",accuracy_score(y_pred,y_test)*100,"%")
cm = confusion_matrix(y_pred,y_test)
print(cm)


# In[67]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




