import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk

docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]
# Initialize a CountVectorizer to use NLTK's tokenizer instead of its 
#    default one (which ignores punctuation and stopwords). 
# Minimum document frequency set to 1. 

fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# .fit_transform does two things:
# (1) fit: adapts fooVzer to the supplied text data (rounds up top words into vector space) 
# (2) transform: creates and returns a count-vectorized output of docs
docs_counts = fooVzer.fit_transform(docs)

# fooVzer now contains vocab dictionary which maps unique words to indexes
fooVzer.vocabulary_