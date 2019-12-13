import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time
from tqdm import tqdm
tqdm.pandas()
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Activation, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.layers import GRU, LSTM
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger

# token_dir = "./preprocess_jobs/text_rm_stopword_keep!?/"
token_dir = sys.argv[1]

train_tokens = pd.read_csv(os.path.join(token_dir, "train_tokens.csv"))
test_tokens = pd.read_csv(os.path.join(token_dir, "test_tokens.csv"))

train_tokens['tokens'] = train_tokens['tokens'].progress_apply(eval)
test_tokens['tokens'] = test_tokens['tokens'].progress_apply(eval)

# %%
total_corpus = train_tokens['tokens'].values.tolist() + test_tokens['tokens'].values.tolist()

#%%
EMB_DIM = 200
W2V_ITERS = 60
W2V_WINDOWS = 10
W2V_MIN_CNT = 2

# MAX_SEQ_LEN = 48
# BATCH_SIZE = 64

print("start training w2v model...", flush=True)

w2v_model = Word2Vec(min_count=W2V_MIN_CNT,
                     window=W2V_WINDOWS,
                     size=EMB_DIM,
                     sample=6e-5, #1e-5 is too small 
                     alpha=0.025, 
                     min_alpha=0.0005, 
                     negative=10,
                     workers=mp.cpu_count())

t = time()
w2v_model.build_vocab(total_corpus, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(total_corpus, total_examples=w2v_model.corpus_count, epochs=W2V_ITERS, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)), flush=True)


# t = time()
# w2v_model = Word2Vec(sentences=total_corpus,
                     # size=EMB_DIM,
                     # window=W2V_windows,
                     # iter=W2V_ITERS,
                     # negative=10,
                     # workers=mp.cpu_count(),
                     # min_count=W2V_MIN_CNT,
                     # compute_loss=True,
                     # sample=1e-5,
                     # alpha=0.025,
                     # min_alpha=0.0005)

# print('Time to train w2v: {} mins'.format(round((time() - t) / 60, 3)), flush=True)

w2v_model.save("w2v_model.bin")

#%%
def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

print(most_similar(w2v_model, ["shit", "fuck", "ass", "bitch", "idiot"])[["shit", "fuck", "ass", "bitch"]])

print(most_similar(w2v_model, ["love", "happy", "not"])[["love", "happy", "not"]])



# %%
# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in total_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))
print(f'Average tweet length: {avg_length / float(len(total_corpus)):.2f}')
print('Max tweet length: {}'.format(max_length))


