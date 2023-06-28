#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\rakesh\\Downloads\\stress.csv", encoding="latin-1")

df


# In[2]:


df.isnull().sum()


# In[3]:


import nltk
import re
from nltk.corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer= nltk. SnowballStemmer("english")
stopword=set (stopwords.words ('english'))
def clean(text):
    text= str(text). lower() #returns a string where all characters are Lower case. Symbols and Numbers are ignored text re. sub('\[-?\],,text) #substring and returns a string with replaced values.
    text=re.sub('\[.*?\]',' ',text)
    text =re. sub('https?://\S+/www\. \S+',' ', text) #whitespace char with pattern text re. sub('<. "?>+,, text) #special char enclosed in square brackets
    text =re. sub('<. "?>+',' ', text)
    text =re. sub( '[%s]' % re. escape(string. punctuation),' ', text) #eliminate punctuation from string text re. sub(' \n',, text)
    text =re. sub('\n',' ', text)
    text = re. sub(' \w\d\w*',' ', text) #word character ASCII punctuation
    text= [word for word in text. split(' ') if word not in stopword] #removing stopwords
    text=" ". join(text)
    text = [stemmer. stem (word) for word in text. split(' ') ]#remove morphological affixes from words
    text=" " .join(text)
    return text
df ["text"] = df ["text"]. apply (clean)


# In[4]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
text=" ".join(i for i in df.text)
stopwords=set(STOPWORDS)
wordcloud= WordCloud ( stopwords=stopwords, background_color="white") . generate (text)
plt. figure(figsize=(10, 10)) 
plt. imshow(wordcloud, interpolation='bilinear' ) 
plt.axis("off")
plt.show()


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
x=np.array (df["text"])
y=np.array (df["label"])
cv=CountVectorizer()
X=cv.fit_transform(x)
print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=34)



# In[14]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)


# In[26]:


user=input("Enter text:")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)


# In[ ]:




