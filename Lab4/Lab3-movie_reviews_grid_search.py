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
print("multinomialBC accuracy: ", np.mean(predicted == y_test))

from sklearn import metrics
# Additional information
print(metrics.classification_report(y_test, predicted, target_names=movie.target_names))

# Confusion matrix
print(metrics.confusion_matrix(y_test, predicted))


# PARAM TUNING USING GRID SEARCH

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect_ngram_range': [(1, 1), (1, 2)],
    'tfidt_use_idf': (True, False),
    'clf_alpha': (1e-2, 1e-3),
}


gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train[:20], y_train[:20])

#print(movie.target_names[gs_clf.predict(['two'])[0]])
