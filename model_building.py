import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ParameterGrid

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(model_type='LSTM', vocab_size=10000, embedding_dim=100, max_length=500, units=64, dropout_rate=0.2, learning_rate=0.001):
    """Dynamically create an LSTM or GRU model based on parameters."""
    if model_type not in ['LSTM', 'GRU']:
        logging.error("Invalid model_type specified. Must be 'LSTM' or 'GRU'.")
        raise ValueError("model_type must be 'LSTM' or 'GRU'")
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)
    ])
    
    if model_type == 'LSTM':
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(units / 2)))
    elif model_type == 'GRU':
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(int(units / 2)))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    logging.info(f"Model created with {model_type}, {units} units, {dropout_rate} dropout rate, {learning_rate} learning rate.")
    
    return model

def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs=10, batch_size=64):
    """Train the model and evaluate it on the test set with enhanced logging."""
    logging.info("Starting model training...")

    # Directory for saving models
    model_dir = './src/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'best_model.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    model.load_weights('best_model.h5')
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    logging.info(f"Model evaluation complete. Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    return history, test_loss, test_accuracy


def hyperparameter_tuning(x_train, y_train, x_val, y_val, x_test, y_test, param_grid):
    """Enhanced grid search to find the best model configuration."""
    best_accuracy = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        logging.info(f"Testing model with parameters: {params}")

        # Extract training-specific parameters and remove them from params
        batch_size = params.pop('batch_size', 64)  # Default to 64 if not specified
        epochs = params.pop('epochs', 10)  # Default to 10 if not specified

        # Create model without training-specific parameters
        model = create_model(**params)

        # Now, train and evaluate the model using the extracted parameters
        history, test_loss, test_accuracy = train_and_evaluate_model(
            model, x_train, y_train, x_val, y_val, x_test, y_test, epochs=epochs, batch_size=batch_size
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = params  # Note: params here doesn't include batch_size and epochs anymore
            best_model = model
            logging.info("New best model found.")

    logging.info(f"Best model parameters: {best_params}, with batch_size={batch_size} and epochs={epochs}")
    logging.info(f"Best model accuracy: {best_accuracy}")
    return best_model
