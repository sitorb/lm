# LSTM-Based Quote Generator

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to generate text, specifically quotes. The model is trained on a dataset of quotes and learns to predict the next character in a sequence based on the preceding characters.

## Code Structure and Logic

1. **Data Loading:**
   - `count_quotes`: Counts the total number of quotes in the dataset.
   - `load_quotes_in_batches`: Loads quotes from a JSON file in batches to handle large datasets efficiently.
2. **Data Preparation:**
   - `data_generator_from_list`: Creates a data generator that yields batches of input sequences and their corresponding target characters for training.
   - `Tokenizer`: Converts text into numerical sequences for the model to process.
3. **Model Building:**
   - `Sequential`: Defines a linear stack of layers for the LSTM model.
   - `Embedding`: Maps input characters to dense vectors.
   - `LSTM`: The core recurrent layer that captures long-range dependencies in the text.
   - `Dropout`: Prevents overfitting by randomly dropping out units during training.
   - `Dense`: Output layer with a softmax activation for predicting the next character.
4. **Training:**
   - `model.compile`: Configures the model for training with the Adam optimizer and categorical cross-entropy loss.
   - `model.fit`: Trains the model on the training data for a specified number of epochs.
5. **Saving and Loading:**
   - `model.save`: Saves the trained model to a file for later use.
   - `load_model`: Loads a saved model.
6. **Evaluation:**
   - `model.evaluate`: Evaluates the model's performance on a validation set.
7. **Visualization:**
   - `plot_training_history`: Plots the training loss and accuracy over epochs.
8. **Text Generation:**
   - `generate_text`: Uses the trained model to generate new text based on a seed text.

## Technology and Algorithms

- **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) architecture well-suited for sequence learning tasks like text generation. LSTMs have memory cells that can store information over long periods, allowing them to capture dependencies in sequential data.
- **TensorFlow/Keras:** A popular deep learning framework used for building and training neural networks.
- **ijson:** A library for efficiently parsing JSON files, used here for loading the quotes dataset.
- **Numpy:** Provides numerical computing capabilities, used for array operations.
- **Matplotlib:** Used for creating visualizations, such as the training progress plot.
- **Scikit-learn:** Used for splitting the data into training and validation sets.


## How it Works

The LSTM model learns the patterns and relationships between characters in the training data. It then uses this knowledge to predict the probability of the next character given a sequence of previous characters. By repeatedly sampling from these probabilities, the model generates new text that resembles the style and content of the training data.

## Usage

To run the project, you need to have Python and the required libraries installed. You can then execute the code in a Jupyter Notebook or Google Colab environment.

## Potential Improvements

- Experiment with different LSTM architectures and hyperparameters to improve the quality of generated text.
- Increase the size of the training dataset to enhance the model's ability to capture diverse patterns.
- Implement techniques like beam search to generate more coherent and grammatically correct text.
