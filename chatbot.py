# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:28:08 2019

@author: Bhama
"""

import nltk
import numpy as np
import random
import string

f=open('D:/bhama krishna/chatbot.txt','r',errors='ignore')

raw=f.read()

 #printing data from the file
 #print(raw)

nltk.download("punkt")
nltk.download("wordnet")

#converts to list of sentences
sent_tokens=nltk.sent_tokenize(raw)

#converts to list of words
word_tokens=nltk.word_tokenize(raw)

#printing sentences
print(sent_tokens[:2])

#printing first two words
print(word_tokens[:2])


""" preprocessing the raw text"""
lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#keyword matching           
GREETING_INPUTS=("hello", "hi", "greetings", "sup", "what's up","hey")

GREETING_RESPONSES=("hi", "hey", "*nods*", "hi there", "hello","Namaste","Vanakkam", "I am glad! You are talking to me")

   
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print('PLUTO: My name is Pluto. I will answer your queries about Chatbots. If you want to exit, type Bye!')

while(flag==True):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flat=False
            print('PLUTO: You are welcome..')
        else:
            if(greeting(user_response)!=None):
                print("PLUTO:"+greeting(user_response))
            else:
                print("PLUTO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("PLUTO: Bye! take care..")
                
            
        
            



