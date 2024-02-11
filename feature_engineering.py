from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Additionnal ffeature engineering to improve model accuracy 

def apply_tfidf_to_text(data, max_features=1000):
    """Apply TF-IDF vectorization to the given text data."""
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_result = tfidf.fit_transform(data).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
    return tfidf_df


