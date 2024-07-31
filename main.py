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
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from sklearn.metrics import accuracy_score

# Load CSV file (using the path)
df = pd.read_csv('/u/alenad_guest/Desktop/finalres/processed_resume.csv')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df_dropped = df.dropna()
    X = df.drop('Category', axis=1).values  # Adjust 'target' to actual target column name
    y = df['Category'].values
    return X, y

# Understanding the dataset
print("First few rows of the DataFrame:")
print(df.head())

# Information about the dataframe
print("\nDataFrame info:")
print(df.info())

# Download NLTK data
nltk.download('wordnet')

# Function to remove punctuation and clean the text
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Convert text to lowercase
    text = remove_punctuation(text)  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize words
    return text

df['processed_text'] = df['Resume_str'].astype(str).apply(preprocess_text)

# Define stop words list
stop_words = [
    'february', 'march', 'april', 'may', 'june', 'by', 'gpa', 'well', 'as', 'that', 'for', 'on', 'or', 'summary',
    'name', 'july', 'august', 'september', 'october', 'november', 'december', 'all', 'new', 'employee', 'i', 'any',
    'employees', 'including', 'also', 'months', 'years', 'worked', 'provided', 'streamlined', 'company', 'city',
    'which', 'job', 'state', 'annual', 'him', 'it', 'made', 'the', 'from', '1990', '1991', '1992', '1993', '1994',
    '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
    '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
    '2023', '2024', 'move', 'someone', 'their', 'every', 'everything', 'at', 'when', 'but', 'if', 'he', 'his', 'is',
    'an', 'to', 'with', 'has', 'in', 'will', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov',
    'dec', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'needs', 'work', 'january', 'united states', 'day',
    'tech mahindra', 'mumbai', 'maharashtra', 'bhopal', 'madhya pradesh', 'rpgv', 'delhi', 'pune', 'less', 'india private'
]


# Function to generate word cloud
def generate_word_cloud(category, texts):
    if not texts.empty:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words, ngram_range=(2, 3))
        X_tfidf = tfidf_vectorizer.fit_transform(texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Trying to understand the frequency of each word 
        word_freq = X_tfidf.sum(axis=0).A1
        word_freq_dict = dict(zip(feature_names, word_freq))

        wc = WordCloud(width=1000, height=500, max_words=50, background_color='white', stopwords=stop_words).generate_from_frequencies(
            dict(zip(feature_names, X_tfidf.sum(axis=0).A1))
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'Word Cloud for {category}')
        plt.axis('off')
        plt.savefig(f'word_cloud_{category}.png')
        plt.show()
        plt.close()
    else:
        print(f"No texts found for category: {category}")

# Word cloud for each label
categories = df['Category'].unique()
for category in categories:
    texts = df[df['Category'] == category]['processed_text']
    generate_word_cloud(category, texts)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['processed_text'])

print("Tokenizer Word Index:", tokenizer.word_index)
print("Vocabulary Size:", len(tokenizer.word_index) + 1)

# Padding 
sequences = tokenizer.texts_to_sequences(df['processed_text'])
padded_sequences = pad_sequences(sequences)

# Store padded sequences in df
df['padded_text'] = list(padded_sequences)

# Extract padded sequences, labels
X = np.array(padded_sequences, dtype=np.float32)
y = df['Category']  # 'Category' is the label column

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = y_encoded.reshape(-1, 1)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_one_hot = one_hot_encoder.fit_transform(y_encoded)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
def create_model(learning_rate):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_one_hot.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 64, 128, 256)
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    
    # Create and train the model
    model = create_model(learning_rate)
    
    # Convert data to TensorFlow tensors if needed
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

    # Training with validation
    history = model.fit(
        X_train_tensor, y_train_tensor, 
        batch_size=batch_size, 
        epochs=num_epochs, 
        verbose=2,
        validation_data=(X_val_tensor, y_val_tensor)
    )

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val_tensor, y_val_tensor, verbose=2)
    
    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    return val_accuracy  # Return validation accuracy for Optuna to maximize

# Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train and save the best model
best_model = create_model(study.best_params['learning_rate'])
best_model.fit(
    tf.convert_to_tensor(X_train, dtype=tf.float32), 
    tf.convert_to_tensor(y_train, dtype=tf.float32),
    batch_size=study.best_params['batch_size'],
    epochs=study.best_params['num_epochs'],
    validation_data=(tf.convert_to_tensor(X_val, dtype=tf.float32), tf.convert_to_tensor(y_val, dtype=tf.float32))
)
best_model.save('best_resume_model.h5')