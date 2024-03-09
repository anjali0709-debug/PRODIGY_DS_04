#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install spacy


# In[28]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import spacy
import re
import string

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import optimizers, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# In[31]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[32]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
nlp = spacy.load('en_core_web_sm')


# In[35]:


get_ipython().system('pip install wget')
import wget
url = 'https://cocl.us/new_york_dataset'
filename = wget.download(url)
print(filename)


# In[36]:


get_ipython().system('wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip')
get_ipython().system('unzip -q glove.6B.zip')


# In[37]:


# Get the GloVe into our directory
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


# In[38]:


train = pd.read_csv("C:/Users/alshi/OneDrive/Documents/twitter_training.csv", index_col=0, header=None, names=['entity', 'label', 'text'])
test = pd.read_csv("C:/Users/alshi/OneDrive/Documents/twitter_validation.csv", index_col=0, header=None, names=['entity', 'label', 'text'])


# In[44]:


def lowercase(data):
    return data['text'].str.lower()

def change_punctuation(data):
    return data['text'].str.replace('`', "'")

def remove_numbers(data):
    return data['text'].replace('[^a-zA-z.,!?/:;\"\'\s]', '', regex=True)

def remove_special_characters(data):
    return data['text'].replace('[^a-zA-Z0-9 ]', '', regex=True)

def custom(data):
    return data['text'].replace('im', 'i am')

def lemmatize(data):
    lemmatized_array = []
    
    for text in data['text']:
        lemmatized_text = []
        doc = nlp(text)
        for token in doc:
           lemmatized_text.append(token.lemma_)
        lemmatized_array.append(' '.join(lemmatized_text))
    return lemmatized_array

def stop_words(data):
    stop_words_array = []
    for text in data['text']:
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        stop_words_array.append(' '.join(filtered_tokens))
    return stop_words_array

def delete_links(data):
    return data['text'].replace(r'http\S+', '', regex=True)

def preprocessing(data):
    df = data.copy()
    df['text'] = lowercase(df)
    df['text'] = custom(df)
    df['text'] = change_punctuation(df)
    df['text'] = lemmatize(df)
    df['text'] = remove_numbers(df)
    df['text'] = delete_links(df)
    df['text'] = remove_special_characters(df)
    return df


# In[45]:


# As seen in dataset, the first entry itself contains many multiple words
train.drop_duplicates(subset=['text'], inplace=True)
train.reset_index(inplace=True)
train['text'] = train['text'].astype('str')
test['text'] = test['text'].astype('str')


# In[46]:


len(train['text'])
len(test['text'])


# In[47]:


train = preprocessing(train)
test = preprocessing(test)


# In[50]:


le = LabelEncoder()
train['label'] = le.fit_transform(train['label'])
test['label'] = le.transform(test['label'])

X = train['text']
y = train['label']


# In[51]:


max_words = 10000
maxlen = 200
emb_dim = 50
training_samples = int(len(X) * 0.8)

text_dataset = tf.data.Dataset.from_tensor_slices(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


max_features = 20000
embedding_dim = 128

vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_words, # Max number of word in the internal dictionnary. We keep the most frequent
        output_mode='int',
        output_sequence_length=maxlen  # Size max of text
        )

vectorize_layer.adapt(text_dataset.batch(64))  


# In[53]:


voc = vectorize_layer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))


# In[54]:


num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# In[55]:


embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])


# In[56]:


model = keras.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    embedding_layer,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax'),
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[64]:


# Time to cook the model...Use GPU if you have low amount of patient
cl = [tf.keras.callbacks.EarlyStopping(
                  monitor='val_accuracy',
                  restore_best_weights=True,
                  patience=10)] 

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=64, callbacks = cl)


# In[58]:


history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()


# In[59]:


predictions = model.predict(test['text'])


# In[60]:


test['label'] = le.fit_transform(test['label'])


# In[61]:


predicted_labels = []

for predictions_array in predictions:
    predicted_labels.append(np.argmax(predictions_array))


# In[62]:


from sklearn.metrics import accuracy_score

accuracy_score(predicted_labels, test['label'])


# In[63]:


print(predicted_labels)


# In[65]:


print(test['label'])


# In[ ]:




