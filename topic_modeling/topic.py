import urllib.request
from sklearn.decomposition import LatentDirichletAllocation
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

def Scaper(url,class_):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,'html.parser')
    scraped_data = soup.find('div',class_=class_).descendants
    data = []
    for i in scraped_data:
        data.append(i.string)
    data.append(i.string)
    data = [x for x in data if x is not None]
    data =[x for x in data if x !="\n"]
    data = list(dict.fromkeys(data))
    for i in data:
        x = i.split(' ')
        if len(x) <=3:
             data.remove(i)
    data.remove(data[1])
    data.remove(data[-1])
    return data


def vectoriser(url,class_,max_df,min_df,no_features):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(Scaper(url,class_))
    vector = tf_vectorizer.get_feature_names()
    return vector,tf

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic = " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append(topic)
    return topics
                

def topic_generation(no_topics,no_top_words,my_list,tf):
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    return display_topics(lda, my_list, no_top_words)
    



TP= "https://www.tutorialspoint.com/computer_programming/computer_programming_syntax.htm"
AN = 'https://www.tutorialspoint.com/computer_programming/computer_programming_data_types.htm'
CLASS= 'col-md-7 middle-col'



data,tf = vectoriser(TP,CLASS,1,3,120)
topic = topic_generation(5,20,data,tf)


data1,tf1 = vectoriser(AN,CLASS,1,3,120)
topics = topic_generation(5,20,data1,tf1)


