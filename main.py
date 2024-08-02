import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout, GlobalMaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
import optuna
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Load CSV file
df = pd.read_csv('/u/alenad_guest/Desktop/finalres/processed_resume.csv')

# Data Preprocessing
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize words
    return text

df['processed_text'] = df['Resume_str'].astype(str).apply(preprocess_text)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['processed_text'])
sequences = tokenizer.texts_to_sequences(df['processed_text'])
padded_sequences = pad_sequences(sequences)

# Define vocabulary size, max length
vocab_size = len(tokenizer.word_index) + 1
max_length = padded_sequences.shape[1]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Category'])
y_encoded = y_encoded.reshape(-1, 1)

# One-hot encode 
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_one_hot = one_hot_encoder.fit_transform(y_encoded)
num_classes = y_one_hot.shape[1]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y_one_hot, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Best parameters from Optuna trial 86
embedding_dim = 100
num_filters = 182
kernel_size = 4
dense_units = 196
dropout_rate = 0.4729
activation = 'tanh'
optimizer = RMSprop(learning_rate=0.0009566)
weight_init = tf.keras.initializers.HeNormal()

# Model creation
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation, kernel_initializer=weight_init))
model.add(GlobalMaxPooling1D())
model.add(Dense(dense_units, activation=activation))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))  # Adjusting for one-hot encoded labels

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',  # Change loss to categorical_crossentropy for one-hot encoded labels
              metrics=['accuracy'])

# Train the model
model.fit(X_train_resampled, y_train_resampled, validation_data=(X_val, y_val), epochs=30, batch_size=64)

# Save the model
model.save('bestresumemodel_trial86.keras')
