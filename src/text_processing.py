"""
Text tokenization and embedding functions
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from IPython.display import display

# Download resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# Text processing
def tokenized_text(text_list):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_data = []
    
    for text in text_list:
        tokens = word_tokenize(text)
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        cleaned_data.append(filtered_tokens)
    return cleaned_data