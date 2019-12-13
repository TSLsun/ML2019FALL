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


# %%
# data_dir = "./data"
# data_dir = sys.argv[1]

# train_x = os.path.join(data_dir, "train_x.csv")
# train_y = os.path.join(data_dir, "train_y.csv")
# test_x = os.path.join(data_dir, "test_x.csv")

train_x = os.path.join(sys.argv[1])
train_y = os.path.join(sys.argv[2])
test_x = os.path.join(sys.argv[3])

x_train = pd.read_csv(train_x)["comment"].values
x_test = pd.read_csv(test_x)["comment"].values
y_label = pd.read_csv(train_y)["label"].values

df = pd.DataFrame(data={'text': x_train, 'label': y_label})
# %%
df.head()

# %%
nlp = spacy.load('en_core_web_sm')

# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
# neg_stopwords = {
#     "against", "although", 'again', 'but', 'cannot', 'enough',
#     'even', 'except', 'few', 'however', 'less', "n't", 'neither',
#     'never', 'nevertheless', 'no', 'nobody', 'none', 'noone',
#     'nor', 'not', 'nothing', 'n‘t', 'n’t', 'only', 'rather',
#     'though', 'unless', 'whatever', 'whether', 'which',
#     'while', 'whither', 'who', 'whoever', 'whole', 'whom',
#     'whose', 'why', 'without', 'yet'
# }
# stopwords = spacy_stopwords - neg_stopwords
# print(stopwords)

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

# %%
df['tokens'] = df['text'].progress_apply(process)
df.to_csv("train_tokens.csv", index=False)

# %%
test_df = pd.DataFrame(data={"text": x_test})
test_df['tokens'] = test_df['text'].progress_apply(process)
test_df.to_csv("test_tokens.csv", index=False)

# %%
print("Finish tokenize...", flush=True)
