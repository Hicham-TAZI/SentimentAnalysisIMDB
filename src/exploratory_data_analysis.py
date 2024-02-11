import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_review_length_distribution(sequences):
    """Plot the distribution of review lengths."""
    review_lengths = [len(seq) for seq in sequences]
    plt.figure(figsize=(10, 6))
    sns.histplot(review_lengths, bins=30)
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    plt.show()

def plot_sentiment_distribution(labels):
    """Plot the distribution of sentiments."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.ylabel('Count')
    plt.show()

def generate_word_cloud(text, title):
    """Generate a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(" ".join(text))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()
