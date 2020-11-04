import sklearn
from sklearn.datasets import load_files

moviedir = '/Users/marcus/Documents/TNM108/TNM108Labs/Lab4/movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk

docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]

fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

docs_counts = fooVzer.fit_transform(docs)

# fooVzer now contains vocab dictionary which maps unique words to indexes
fooVzer.vocabulary_

# this vector is small enough to view in a full, non-sparse form!
docs_counts.toarray()

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
fooTfmer = TfidfTransformer()

# Again, fit and transform
docs_tfidf = fooTfmer.fit_transform(docs_counts)
docs_tfidf.toarray()

# A list of new documents
newdocs = ["I have a rose and a lily.", "What a beautiful day."]

# This time, no fitting needed: transform the new docs into count-vectorized form
# Unseen words ('lily', 'beautiful', 'have', etc.) are ignored
newdocs_counts = fooVzer.transform(newdocs)
#print(newdocs_counts.toarray())

# Again, transform using tfidf
newdocs_tfidf = fooTfmer.transform(newdocs_counts)
#print(newdocs_tfidf.toarray())

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target,
                                                          test_size = 0.20, random_state = 12)

# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
print(movieVzer.vocabulary_.get('screen'))

# Likewise, Mr. Steven Seagal is present...
print(movieVzer.vocabulary_.get('seagal'))

# huge dimensions! 1,600 documents, 3K unique terms.
docs_train_counts.shape

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
print('docs_train_tfidft.shape', docs_train_tfidf.shape)

# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Now ready to build a classifier.
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
print('sklearn.metrics.accuracy_score(y_test, y_pred)', sklearn.metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('cm', cm)

# Fake reviews!

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
            'Steven Seagal was terrible', 'Steven Seagal shone through.',
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))


