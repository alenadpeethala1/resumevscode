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

# Define stop words list
from nltk.corpus import stopwords

nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = set([
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
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'needs', 'work', 'january', 'united states', 'day'
])

stopword_list = nltk_stopwords.union(custom_stopwords)

def remove_stopwords(texts):
    return [' '.join([word for word in text.split() if word.lower() not in stopword_list]) for text in texts]

# Function to generate word cloud
def generate_word_cloud(category, texts):
    if not texts.empty:
        processed_texts = remove_stopwords(texts)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(2, 3), stop_words=list(stopword_list))
        X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Create a dictionary of word frequencies
        word_freq = X_tfidf.sum(axis=0).A1
        word_freq_dict = dict(zip(feature_names, word_freq))

        wc = WordCloud(width=1000, height=500, max_words=50, background_color='white').generate_from_frequencies(word_freq_dict)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'Word Cloud for {category}')
        plt.axis('off')
        plt.savefig(f'word_cloud_{category}.png')
        plt.show()
        plt.close()
    else:
        print(f"No texts found for category: {category}")


# Generate word clouds for each label
categories = df['Category'].unique()
for category in categories:
    texts = df[df['Category'] == category]['processed_text']
    generate_word_cloud(category, texts)

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
history = model.fit(X_train_resampled, y_train_resampled, validation_data=(X_val, y_val), epochs=30, batch_size=64, verbose=1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_plot.png')  # Save without specifying a path
plt.show()  # Display the plot
plt.close()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_plot.png')  # Save without specifying a path
plt.show()  # Display the plot
plt.close()

# Save the model
model.save('bestresumemodel_trial86.keras')