# Text Similarity and Sentiment Analysis with LSTM/GRU

This project aims to demonstrate the application of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models in performing sentiment analysis on the IMDB movie reviews dataset. It showcases the power of recurrent neural networks in understanding text data, capturing temporal dependencies, and classifying sentiments as positive or negative. Additionally, the project includes hyperparameter tuning to find the optimal model configuration and exploratory data analysis (EDA) to gain insights into the dataset.

## Features

- Sentiment analysis using LSTM and GRU models.
- Hyperparameter tuning for model optimization.
- Exploratory data analysis (EDA) including review length distribution, sentiment distribution, and word clouds.
- Structured logging for monitoring model training and evaluation processes.
- Command-line interface for easy switching between EDA, training, and evaluation.

## Project Structure


I understand your request now. Here's the entire README content consolidated into one markdown block that you can use directly in your README.md file:

markdown
Copy code
# Text Similarity and Sentiment Analysis with LSTM/GRU

This project demonstrates the application of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models for sentiment analysis on the IMDB movie reviews dataset. It highlights the power of recurrent neural networks to understand text data, capture temporal dependencies, and classify sentiments as positive or negative. Additionally, this project includes hyperparameter tuning for optimal model configuration and exploratory data analysis (EDA) to derive insights from the dataset.

## Features

- Sentiment analysis using LSTM and GRU models
- Hyperparameter tuning for model optimization
- Exploratory data analysis (EDA) including review length distribution, sentiment distribution, and word clouds
- Structured logging for monitoring the model training and evaluation processes
- Command-line interface for easy toggling between EDA, training, and evaluation

## Project Structure

```plaintext
TextSimilarityLSTM/
│
├── src/
│   ├── data_preprocessing.py       # Script for data preprocessing
│   ├── exploratory_data_analysis.py# Script for EDA
│   ├── feature_engineering.py      # Script for feature extraction and engineering
│   ├── model_building.py           # Script for model creation and hyperparameter tuning
│   └── main.py                     # Main script to run the project
│
├── hyperparams.json                # JSON file containing hyperparameters for tuning
├── requirements.txt                # Python dependencies required for the project
└── README.md                       # Documentation of the project

```


## Setup

1. **Clone the Repository**
    ```
    git clone https://github.com/Hicham-TAZI/SentimentAnalysisIMDB.git
    cd TextSimilarityLSTM
    ```

2. **Create and Activate a Virtual Environment**
    - For Windows:
        ```
        python -m venv env
        .\env\Scripts\activate
        ```
    - For macOS/Linux:
        ```
        python3 -m venv env
        source env/bin/activate
        ```

3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

## Usage

- **Perform EDA**
    ```
    python main.py --action eda
    ```

- **Train the Model**
    ```
    python main.py --action train
    ```
    This will read hyperparameters from `hyperparams.json` and perform hyperparameter tuning.

- **Evaluate the Model**
    ```
    python main.py --action evaluate
    ```
    Ensure to load the best model saved during training for evaluation.

## Customization

Modify `hyperparams.json` to experiment with different model configurations and training settings. Update the `create_model` function in `model_building.py` for experimenting with different neural network architectures.

## Contributing

Contributions to improve the project are welcome. See `CONTRIBUTING.md` for how to help.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, feature requests, or contributions, please open an issue in the GitHub issue tracker for this repository.
