from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf',MultinomialNB()),
])

import sklearn
from sklearn.datasets import load_files

moviedir = '/Users/marcus/Documents/TNM108/TNM108Labs/Lab4/movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.20, random_state = 12)

text_clf.fit(docs_train, y_train)

import numpy as np
predicted = text_clf.predict(docs_test)
print("multinomialBCaccuracy", np.mean(predicted == y_test))

