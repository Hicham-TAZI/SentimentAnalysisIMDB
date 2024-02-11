import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure you have downloaded the necessary NLTK data with nltk.download('stopwords') and nltk.download('wordnet')

def clean_text(text):
    """Function to clean text by removing punctuation, stopwords, and performing lemmatization."""
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)

def preprocess_data(filepath, vocab_size=10000, max_length=500, padding_type='post', truncating_type='post'):
    """Load data from a CSV file, clean, and tokenize text."""
    df = pd.read_csv(filepath)
    df['cleaned_text'] = df['review'].apply(clean_text)
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncating_type)
    
    return padded_sequences, df['sentiment'].values, tokenizer
