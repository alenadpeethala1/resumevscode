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
from tensorflow.keras.optimizers import Adam
import optuna
from sklearn.metrics import accuracy_score
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

def objective(trial):
    # Hyperparameters to tune
    embedding_dim = trial.suggest_categorical('embedding_dim', [50, 100, 200])
    num_filters = trial.suggest_int('num_filters', 32, 256)
    kernel_size = trial.suggest_int('kernel_size', 3, 7)
    dense_units = trial.suggest_int('dense_units', 64, 512)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_init = trial.suggest_categorical('weight_init', ['glorot_uniform', 'he_normal', 'lecun_normal'])

    # Create model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=dense_units, activation=activation, kernel_initializer=weight_init))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile model
    opt = tf.keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Learning Rate Scheduler
    def lr_schedule(epoch, lr):
        decay = 0.1
        min_lr = 1e-6
        if epoch > 10:
            new_lr = max(lr * tf.math.exp(-decay), min_lr)
            return float(new_lr)
        return float(lr)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=trial.suggest_int('num_epochs', 10, 50),
        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
        callbacks=[lr_scheduler, early_stopping],
        verbose=0
    )

    # Get validation acc
    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

# Optimize hyperparameters - Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters found: ", study.best_params)
