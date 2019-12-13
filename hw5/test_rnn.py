#%%
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from time import time
from tqdm import tqdm
tqdm.pandas()
import re
import string
import spacy
import emoji
from itertools import groupby
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Activation, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.layers import GRU, LSTM
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# %%
test_csv = sys.argv[1]
pred_csv = sys.argv[2]
w2v_model_dir = sys.argv[3]
model_dir = sys.argv[4]
# model_dir = "w2v_rnn_max_val_acc.h5"

x_test = pd.read_csv(test_csv)["comment"].values

nlp = spacy.load('en_core_web_sm')

stopwords = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
             'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
             'it', 'its', 'they', 'them', 'their', 'theirs', 
             'will', 'would', 'a', 'an', 'the', 
             'as', 'if', 'and', 'or', 'this', 'that',
             'until', 'of', 'on', 'in', 'at', 'to', 
             'here', 'for', 'any', 'all', 'between',
             'do', 'does', 'did', 'own', 'has', 'have', 
             'am', 'is', 'are', 'be', 'was', 'were',
             'about', '-pron-', 'say', 'said',
             "'s", "’s", "'re", "’re", "'m", "’m",
            }

def process(tweet):
    ### emoji to text :xxx_face:
    tweet = emoji.demojize(tweet)
#     tweet = tweet.replace(":", " ")
    
    ### remove @user URL
    tweet = tweet.replace("@user", " ")
    tweet = tweet.replace("URL", " ")
    
    ### remove HTML words
    tweet = tweet.replace("&amp;", " ")
    tweet = tweet.replace("&lt;", " ")
    tweet = tweet.replace("&gt;", " ")
    
    ### remove numbers
    tweet = re.sub("[0-9]", " ", tweet)
    
    ### remove punctuation except "!?" (wondering why cannot keep "." as well QQ)
    # punct = string.punctuation
    # punct = punct.replace("?", "").replace("!", "").replace("_", "")
    # punct = punct.replace("'", "").replace("’", "")
    # pattern = f"[{punct}]" # create the pattern 
    # tweet = re.sub(pattern, " ", tweet)
    
    ### remove all punctuation (left only a-z A-Z 0-9)
    tweet = re.sub('[^a-zA-Z0-9]', " ", tweet)
     
    ### remove extra spaces
    tweet = ' '.join(tweet.split())
    
    ### tokenize
    doc = nlp(tweet)
    # tokens = [token.text.lower() for token in doc]
    tokens = [token.lemma_.lower() for token in doc]
    
    ### remove duplicate words
    tokens = [x[0] for x in groupby(tokens)]
    
    ### remove stopwords
    # tokens = [token for token in tokens if token not in stopwords]
   
    ### clean some punct
    # tokens = [ token for token in tokens 
                # if token not in set(string.punctuation) ]
                # if token not in set(string.punctuation) - {"!", "?"} | {"‘", "’", '”', '“'}]
    
#     return tweet
    return tokens


test_df = pd.DataFrame(data={"text": x_test})
test_df['tokens'] = test_df['text'].progress_apply(process)
# test_df.to_csv("test_tokens.csv", index=False)
print("Finish tokenize...", flush=True)


# test_tokens['tokens'] = test_tokens['tokens'].progress_apply(eval)

## PARAM
MAX_SEQ_LEN = 100
EMB_DIM = 200
BATCH_SIZE = 128

### load w2v model
w2v_model = Word2Vec.load(w2v_model_dir)

print("Load w2v model ...", flush=True)

#%%
def rnn_model(numClasses=2):
    input = Input(shape=(MAX_SEQ_LEN, EMB_DIM))
    x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(input)
    x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, activation="tanh", return_sequences=False, dropout=0.3))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)
    return Model(input, output)

model = load_model(model_dir)
print("Load w2v_rnn_max_val_acc.h5 ....", flush=True)

#%% Predict
print("Predict Test data...")
test_texts = test_df['tokens'].values.tolist()

# pad zeros at the end
test_data_emb = np.zeros((len(test_texts), MAX_SEQ_LEN, EMB_DIM))
for n in range(len(test_texts)):
    for i in range(min(len(test_texts[n]), MAX_SEQ_LEN)):
        try:
            vec = w2v_model.wv[test_texts[n][i]]
            test_data_emb[n][i] = (vec - vec.mean(0)) / (vec.std(0) + 1e-10)
        except KeyError as e:
            # print(test_texts[n][i], 'is not in dict.')
            continue

y_test_pred = model.predict(test_data_emb).argmax(axis=-1)

pred_df = pd.DataFrame({'id': np.arange(0, len(y_test_pred)), 'label': y_test_pred})
pred_df.to_csv(pred_csv, index=False)

# print("pred distribution: ")
# print((y_test_pred==0).sum())
# print((y_test_pred==1).sum())

