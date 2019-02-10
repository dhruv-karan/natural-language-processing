import urllib.request

g4g= "https://www.geeksforgeeks.org/dbms-interfaces/"

page = urllib.request.urlopen(g4g)


from bs4 import BeautifulSoup
soup = BeautifulSoup(page,'html.parser')

z =soup.find('h1',class_="entry-title").string
#Z ===> z gives the heading of article


scraped_data = soup.find('div',class_="entry-content").descendants

data = []

for i in scraped_data:
    data.append(i.string)

data = filter(None, data)

data = [x for x in data if x is not None]
data =[x for x in data if x !="\n"]
#=========== data cleaned=============================

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 100

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=100, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data)
tf_feature_names = tf_vectorizer.get_feature_names()




from sklearn.decomposition import NMF, LatentDirichletAllocation

no_topics = 5

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)



def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print (topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)





#=================== we can we either LDA and NMF algorithum for topic generation =================






