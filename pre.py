import nltk
import spacy
import re
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct])
