#Import libraries
import json
import nltk
import re
import string
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
pd.options.mode.chained_assignment = None
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#loading the dataset
data=open("data.json","r")
data=json.loads(data.read())

labels=list(data.keys())
values=list(data.values())


prompt=input("Enter text: ")

#preprocessing the prompt
with open('./abbreviations.json') as json_file:
    abbreviations = json.load(json_file)

#functions 
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


from collections import Counter
cnt = Counter()
for text in prompt:
    for word in text.split():
        cnt[word] += 1


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_abbreviation(text):
    new_text = []
    for w in text.split():
        if w.lower() in list(abbreviations.keys()):
            new_text.append(abbreviations[w.lower()])
        else:
            new_text.append(w)
    return " ".join(new_text)


def clean_text(string):
    string=remove_punctuation(string)
    string=remove_stopwords(string)
    string=remove_freqwords(string)
    string=remove_rarewords(string)
    string=stem_words(string)
    string=lemmatize_words(string)
    string=remove_urls(string)
    string=remove_html(string)
    string=remove_abbreviation(string)
    
    return string

print(clean_text(prompt))