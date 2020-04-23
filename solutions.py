#
# Solutions to working-with-text.ipynb
#

##########################################
# 1
##########################################



def preprocess(s):
    return s.lower()

def search(docs, query):
    query = preprocess(query)
    for doc in docs:
        doc = preprocess(doc)
        if query in doc:
            return doc

##########################################
# 2
##########################################

import re
not_alphanumeric_or_space = re.compile(r'[^\w|\s]')

def word_count(doc):
    counts = {}
    for word in doc.split():
        try:
            counts[word] += 1
        except KeyError:
            counts[word] = 1
    return counts


def score(doc, query):
    tokens = query.split()
    count = word_count(doc)
    count = {k:v for k, v in count.items() if k in tokens}
    return sum(count.values())


def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc)
    return doc.lower()

def search(docs, query):
    docs = [preprocess(doc) for doc in docs]
    query = preprocess(query)
    scores = [score(doc, query) for doc in docs]
    tog = sorted(zip(docs, scores), key=lambda x: x[1])
    doc, _ = tog[-1]
    return doc



##########################################
# 3
##########################################


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
not_alphanumeric_or_space = re.compile('[^(\w|\s|\d)]')

def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc)
    words = [stemmer.stem(word) for word in doc.split()]
    return ' '.join(words).lower()



##########################################
# 4
##########################################


import numpy as np

def inverse_doc_freq(term, docs):
    df = sum([1 for doc in docs if term in doc])
    return np.log(len(docs) / df)

def score(doc, query, docs):
    tokens = query.split()
    count = word_count(doc)
    count = {k:v for k, v in count.items() if k in tokens}
    freqs = [inverse_doc_freq(k, docs) for k, v in count.items()]
    count = {k:v*t for (k,v), t in zip(count.items(), freqs)}
    return sum(count.values())

def search(docs, query):
    docs = [preprocess(doc) for doc in docs]
    query = preprocess(query)

    scores = [score(doc, query, docs) for doc in docs]
    tog = sorted(zip(docs, scores), key=lambda x: x[1])
    doc, _ = tog[-1]
    return doc
