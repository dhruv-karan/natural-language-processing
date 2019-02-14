import urllib.request
from bs4 import BeautifulSoup

    
def scrape(url,id_):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    scraped = soup.find('div',class_=id_).descendants
    data = []
    for i in scraped:
        data.append(i.string)
    data = [x for x in data if x is not None]
    data =[x for x in data if x !="\n"]
    data = list(dict.fromkeys(data))
    for i in data:
        x = i.split(' ')
        if len(x) <=3:
            data.remove(i)
        if type(i)== int or type(i)==float:
            data.remove(i)
    return data

def token_to_sentence(mylist):
    word = ' '
    for i in mylist:
        if type(i)==int:
            print(i)
        else:
            word = word + i + ' '
    return word

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

def tdidf_vec(no_features,url,class_):
    tfidf_vectorizer = TfidfVectorizer(max_df=10, min_df=1, max_features=no_features, stop_words='english')
    tfidf_vectorizer.fit_transform(scrape(url,class_))
    return tfidf_vectorizer.get_feature_names()
def Stem(mylist):
    pst = PorterStemmer()
    for t,i in enumerate(mylist):
        i = pst.stem(i)
        mylist[t] = i
        

TP= "https://www.tutorialspoint.com/computer_programming/computer_programming_environment.htm"
HOME = 'https://www.tutorialspoint.com/computer_programming/index.htm'
BASICS = 'https://www.tutorialspoint.com/computer_programming/computer_programming_overview.htm'
CLASS= "col-md-7 middle-col" # this class is corresponding to the div in url TP

tfidf_feature_names_1 = tdidf_vec(150,HOME,CLASS)
tfidf_feature_names = tdidf_vec(150,TP,CLASS)
tfidf_feature_names_2 = tdidf_vec(150,BASICS,CLASS)

tfidf_feature_names = token_to_sentence(tfidf_feature_names)
tfidf_feature_names_1 = token_to_sentence(tfidf_feature_names_1)
tfidf_feature_names_2 = token_to_sentence(tfidf_feature_names_2)



documents = [tfidf_feature_names,tfidf_feature_names_1,tfidf_feature_names_2]
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)


doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
                  columns=count_vectorizer.get_feature_names())

df


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(df, df)
