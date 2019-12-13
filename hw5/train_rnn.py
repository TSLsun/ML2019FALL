#%%
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
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# %%
# token_dir = './'
token_dir = sys.argv[1]
w2v_model_dir = sys.argv[2]

train_tokens = pd.read_csv(os.path.join(token_dir, "train_tokens.csv"))
test_tokens = pd.read_csv(os.path.join(token_dir, "test_tokens.csv"))

train_tokens['tokens'] = train_tokens['tokens'].progress_apply(eval)
test_tokens['tokens'] = test_tokens['tokens'].progress_apply(eval)

# %%
total_corpus = train_tokens['tokens'].values.tolist() + test_tokens['tokens'].values.tolist()

print(len(total_corpus))

# %%
## PARAM
MAX_SEQ_LEN = 100
EMB_DIM = 200
BATCH_SIZE = 128

### load w2v model
w2v_model = Word2Vec.load(w2v_model_dir)

print("w2v model loaded...", flush=True)


#%%
print('Converting texts to vectors...', flush=True)

texts = train_tokens['tokens'].values.tolist()

# pad zeros at the end
data_emb = np.zeros((len(texts), MAX_SEQ_LEN, EMB_DIM))
for n in range(len(texts)):
    for i in range(min(len(texts[n]), MAX_SEQ_LEN)):
        try:
            vec = w2v_model.wv[texts[n][i]]
            data_emb[n][i] = (vec - vec.mean(0)) / (vec.std(0) + 1e-10)
        except KeyError as e:
            # print(texts[n][i], 'is not in dict.')
            continue

# %%
print(data_emb.shape)

# %%
def train_test_split(X, y, test_size=0.2, random_state=42):
    n_train = int((1 - test_size) * len(X))
    n_test = len(X) - n_train
    n_samples = len(X)

    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]

    X_train = X[ind_train]
    X_valid = X[ind_test]
    y_train = y[ind_train]
    y_valid = y[ind_test]

    return X_train, X_valid, y_train, y_valid

#%%
labels =  to_categorical(train_tokens['label'].values)

X_train, X_valid, Y_train, Y_valid = train_test_split(data_emb, labels, test_size=0.2, random_state=42)

# %%
def rnn_model(numClasses=2):
    input = Input(shape=(MAX_SEQ_LEN, EMB_DIM))
    x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(input)
    x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, activation="tanh", return_sequences=False, dropout=0.3))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)
    return Model(input, output)

model = rnn_model()
print (model.summary())

# opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=3*1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
max_val_acc = ModelCheckpoint("w2v_rnn_max_val_acc.h5", monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)

print('Training model...', flush=True)
# Fit the model.
fitHistory = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=100, 
                        validation_data=(X_valid, Y_valid), 
                        callbacks=[earlyStop, max_val_acc], 
                        verbose=2)

# Save model to h5 file.
model.save('w2v_rnn.h5')

#%%
model = load_model("w2v_rnn_max_val_acc.h5")
print("Load w2v_rnn_max_val_acc.h5 ....")

# %%
hist = pd.DataFrame(fitHistory.history)
hist['epoch'] = fitHistory.epoch
hist.to_csv("fitHistory.csv", index=False)

### plot
plt.figure()
plt.plot(hist['epoch'], hist['loss'], label='train')
plt.plot(hist['epoch'], hist['val_loss'], label='valid')
# plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig("loss_history.png", dpi=100)

plt.figure()
plt.plot(hist['epoch'], hist['acc'], label='train')
plt.plot(hist['epoch'], hist['val_acc'], label='valid')
# plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('Acc')
plt.legend()
# plt.show()
plt.savefig("acc_history.png", dpi=100)


#%% validation data
y_valid_pred = model.predict(X_valid).argmax(axis = -1)
print("Validation data metrics...\n")
print("Classification report for classifier %s:\n%s\n"
    % (model, metrics.classification_report(Y_valid.argmax(axis = -1), y_valid_pred, digits=3)))
print("", flush=True)

#%% Predict
print("Predict Test data...")
test_texts = test_tokens['tokens'].values.tolist()

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


pred_dir = "./"
pred_df = pd.DataFrame({'id': np.arange(0, len(y_test_pred)), 'label': y_test_pred})
pred_df.to_csv(os.path.join(pred_dir, f'pred.csv'), index=False)

# %%
print("pred distribution: ")
print((y_test_pred==0).sum())
print((y_test_pred==1).sum())

# %%
print("\ntrain distribution: ")
print((train_tokens['label'].values==0).sum())
print((train_tokens['label'].values==1).sum())
