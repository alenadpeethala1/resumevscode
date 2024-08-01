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
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import optuna
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# Load CSV file
df = pd.read_csv('/u/alenad_guest/Desktop/finalres/processed_resume.csv')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df_dropped = df.dropna()
    X = df.drop('Category', axis=1).values
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
stop_words = [ 'february', 'march', 'april', 'may', 'june', 'by', 'gpa', 'well', 'as', 'that', 'for', 'on', 'or', 'summary',
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
    'tech mahindra', 'mumbai', 'maharashtra', 'bhopal', 'madhya pradesh', 'rpgv', 'delhi', 'pune', 'less', 'india private']

# Function to generate word cloud
def generate_word_cloud(category, texts):
    if not texts.empty:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words, ngram_range=(2, 3))
        X_tfidf = tfidf_vectorizer.fit_transform(texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Frequency of each word 
        word_freq = X_tfidf.sum(axis=0).A1
        word_freq_dict = dict(zip(feature_names, word_freq))

        wc = WordCloud(width=1000, height=500, max_words=50, background_color='white', stopwords=stop_words).generate_from_frequencies(
            word_freq_dict
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
y = df['Category']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = y_encoded.reshape(-1, 1)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_one_hot = one_hot_encoder.fit_transform(y_encoded)

# Train test split
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Print shapes before and after SMOTE
print(f"Original training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")

# Check class distribution before SMOTE
y_train_classes = np.argmax(y_train, axis=1)
unique_classes, counts_before = np.unique(y_train_classes, return_counts=True)
print("Class distribution before SMOTE:")
for cls, count in zip(unique_classes, counts_before):
    print(f"Class {cls}: {count} samples")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
y_train_resampled_classes = np.argmax(y_train_resampled, axis=1)
unique_classes_resampled, counts_after = np.unique(y_train_resampled_classes, return_counts=True)
print("\nClass distribution after SMOTE:")
for cls, count in zip(unique_classes_resampled, counts_after):
    print(f"Class {cls}: {count} samples")

print(f"Resampled training set shape: X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")

# Define the model
def create_model(learning_rate):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X.shape[1]))  # Reduced output_dim
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))  # Fewer filters
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))  # Fewer units
    model.add(Dropout(rate=0.3))  # Reduced dropout rate
    model.add(Dense(y_one_hot.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    
    model = create_model(learning_rate)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_resampled, 
        y_train_resampled, 
        epochs=num_epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    return val_accuracy

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best hyperparameters found: ", study.best_params)

# Retrain model with best hyperparameters
best_params = study.best_params
model = create_model(best_params['learning_rate'])
history = model.fit(X_train_resampled, y_train_resampled, 
                    epochs=10, 
                    batch_size=best_params['batch_size'], 
                    validation_data=(X_val, y_val))

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true_classes = np.argmax(y_val, axis=1)
accuracy = accuracy_score(y_val_true_classes, y_val_pred_classes)
print(f'Validation Accuracy: {accuracy}')

# Save model
model.save('resume_classification_model.h5')
