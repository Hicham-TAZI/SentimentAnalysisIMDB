import argparse
import logging
import json
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import exploratory_data_analysis as eda
import model_building

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading and preprocessing IMDB dataset...")
    vocab_size = 10000
    max_length = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
    return x_train, y_train, x_test, y_test

def decode_review(text, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded = ' '.join([reverse_word_index.get(i, '?') for i in text])
    return decoded

def load_hyperparams(json_filepath):
    """Load hyperparameters from a JSON file."""
    with open(json_filepath, 'r') as json_file:
        data = json.load(json_file)
    return data['param_grid']

def main():
    parser = argparse.ArgumentParser(description="Text Similarity and Sentiment Analysis Pipeline")
    parser.add_argument('--action', choices=['eda', 'train', 'evaluate'], required=True, help="Action to perform: eda, train, evaluate")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Within the main function, update the EDA section

    if args.action == 'eda':
        logging.info("Performing exploratory data analysis...")
        
        # Load the word index
        word_index = imdb.get_word_index()
        
        # Select a subset of reviews to generate word clouds
        # Here, we take the first 1000 reviews for demonstration; adjust as needed
        subset_reviews = x_train[:1000]
        decoded_reviews = [decode_review(review, word_index) for review in subset_reviews]
        
        # Combine all reviews in the subset for word cloud generation
        combined_reviews = " ".join(decoded_reviews)
        
        eda.plot_review_length_distribution(x_train)
        eda.plot_sentiment_distribution(y_train)
        eda.generate_word_cloud(combined_reviews, "IMDB Reviews")

    elif args.action == 'train':
        logging.info("Loading hyperparameters...")
        param_grid = load_hyperparams('hyperparams.json')
        
        x_train, y_train, x_test, y_test = load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        logging.info("Starting training with hyperparameter tuning...")
        best_model = model_building.hyperparameter_tuning(x_train, y_train, x_val, y_val, x_test, y_test, param_grid)


    elif args.action == 'evaluate':
        logging.info("Evaluating the best model...")
        # Load the best model. This requires you to have saved the best model during training.
        # For demonstration, we'll assume the best model is already loaded as `best_model`.
        # In practice, you'd load the model using `tf.keras.models.load_model('path/to/model')`.
        test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=1)
        logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
