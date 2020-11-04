# Text feature extraction

d1 = "The sky is blue"
d2 = "The sun is bright"
d3 = "The sun in the sky is bright"
d4 = "We can see the shining sun, the bright sun"

Z = (d1, d2, d3, d4)

from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()

#print vectorizer

my_stop_words = {"the", "is"}
my_vocabulary = {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer = CountVectorizer(stop_words=my_stop_words, vocabulary=my_vocabulary)

#print(vectorizer.vocabulary)
#print(vectorizer.stop_words)

smatrix = vectorizer.transform(Z)
#print(smatrix)

matrix = smatrix.todense()
print('Dense matrix')
print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2") #l2 norm to normalize
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names()
import pandas as pd
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])

print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

first_document = tf_idf_vector[3]
# print the score
df = pd.DataFrame(first_document.T.todense(), index = feature_names, columns= ["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)

print(df)

# Document samilarity

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names


my_categories = ['rec.sport.baseball', 'rec.motorcycles', 'sci.space', 'comp.graphics']
train=fetch_20newsgroups(subset='train', categories=my_categories)
test=fetch_20newsgroups(subset='test', categories=my_categories)

# Length of the traing- and test set and a sample from one of the training text.
print(len(train.data))
print(len(test.data))
print(train.data[9])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train_tfidf,train.target)

docs_new=['Pierangeloisareallygoodbaseballplayer','Mariarideshermotorcycle','OpenGLontheGPUisfast','Pierangelorideshismotorcycleandgoestoplayfootballsinceheisagoodfootballplayertoo.']
X_new_counts=cv.transform(docs_new)
X_new_tfidf=tfidf_transformer.transform(X_new_counts)
predicted=model.predict(X_new_tfidf)
for doc, category in zip(docs_new,predicted):
    print('{!r}=>{}'.format(doc, train.target_names[category]))