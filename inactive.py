import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('path_to_your_model.h5')

# Define the class labels (adjust this list according to your model's classes)
class_labels = ['Engineering', 'Finance', 'Marketing', 'Health', 'Education']

# Define the maximum sequence length (adjust to match your model's configuration)
max_len = 500  # Update this based on your training setup

# Load or define your tokenizer
# Example: Load a tokenizer saved as pickle
import pickle
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the CSV file
df = pd.read_csv('path_to_your_csv_file.csv')

# Inspect the DataFrame
print(df.head())
print(df.columns)

# Preprocess text data from the CSV
texts = df['text_column'].tolist()  # Replace 'text_column' with the actual column name
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Make predictions using your model
predictions = model.predict(padded_sequences)

# Print raw prediction outputs (for debugging)
print("Raw prediction outputs:", predictions)

# Get the index of the highest probability for each prediction
predicted_indices = np.argmax(predictions, axis=1)

# Map indices to class labels
predicted_classes = [class_labels[index] for index in predicted_indices]

# Add predictions to DataFrame
df['Predicted_Class'] = predicted_classes

# Save the results to a new CSV file
df.to_csv('predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'")
