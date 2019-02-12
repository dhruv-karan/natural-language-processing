import urllib.request
from bs4 import BeautifulSoup
TP= "https://www.tutorialspoint.com/computer_programming/computer_programming_environment.htm"
QUORA = 'https://www.quora.com/What-is-the-programming-environment'
CLASS1= "col-md-7 middle-col"
CLASS2= "ExpandedAnswer ExpandedContent"  
    
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



from sklearn.feature_extraction.text import TfidfVectorizer

no_features = 100

tfidf_vectorizer = TfidfVectorizer(max_df=10, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(scrape(TP,CLASS1))
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

no_features = 100

tfidf_vectorizer = TfidfVectorizer(max_df=10, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(scrape(QUORA,CLASS2))
tfidf_feature_names_1 = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity


simlarity = cosine_similarity(tfidf_feature_names,tfidf_feature_names_1)
