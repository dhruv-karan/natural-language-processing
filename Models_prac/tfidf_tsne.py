# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:58:20 2018

@author: dhruv
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from utils import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer

def main():
    sentences, word2idx = get_wikipedia_data(n_files=10,n_vocab=1500, by_paragraph=True)
    with open('w2v_word2idx.json','w') as f:
         json.dump(word2idx,f)
         
    V = len(word2idx)
    N = len(sentences)
    A= np.zeros(V,N)
    j=0
    for sentence in sentences:
        for i in sentence:
            A[i,j] +=1
        j+=1
    print("finished getting raw counts")

transform  = TfidfTransformer()
A = transform.fit_transform(A)
A = A.array()

idx2word = {v:k for k,v in word2idx.iteritems()}

 

        