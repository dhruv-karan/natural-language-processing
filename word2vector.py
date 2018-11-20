# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:13:53 2018

@author: dhruv
"""

import pandas as pd
import numpy as np
import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming')
soup = bs.BeautifulSoup(source,'lxml')
text =""

for paragraph in soup.find_all('p'):
    text += paragraph.text
    
    
text = re.sub(r'\[[0-9]*\]',' ', text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\W',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab
glob = model.wv['global']
relation = model.wv.most_similar('global')