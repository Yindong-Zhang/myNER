from fopen import corpora, Sentences
from gensim.models import TfidfModel,LsiModel
from gensim.corpora import Dictionary
import pandas as pd
import numpy as np
from config import DICTLENGTH, NUMSAMPLES, SVDSIZE

# %%
class TFIDF_corpus():
    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dict = dictionary
        self.tfidf = TfidfModel(dictionary= dictionary)

    def __iter__(self):
        for i, line in enumerate(self.corpus):
            bow = self.tfidf[self.dict.doc2bow(line)]
            # print(i)
            if len(bow) > 0:
                yield bow
            else:
                print("Zero length comment encountered: %s" %(i, ))
                continue

# %%
filepath = "./data/train_input.csv"
corpus = Sentences(filepath, loop=False)
dict = Dictionary(corpus, prune_at=DICTLENGTH)
dict.filter_extremes(no_below= 2, no_above= 0.8)

# %%
for i, bow in enumerate(TFIDF_corpus(corpus, dict)):
    print("---------%s----------" %(i, ))
    print(len(bow))
    print(bow)
    if(i == 20):
        break;
# %%
embed_size = SVDSIZE
lsi = LsiModel(TFIDF_corpus(corpus, dict), num_topics= embed_size)
comments_embd = lsi[TFIDF_corpus(corpus, dict)]

# %%
for i, embd in enumerate(comments_embd):
    print(i, len(embd))

# %%
labels = pd.read_csv('./data/train_input.csv', usecols = ['label', ], squeeze = True)

x = np.zeros((NUMSAMPLES, embed_size))
y = np.zeros((NUMSAMPLES, 1))

count = 0
for i, (embed, l) in enumerate(zip(comments_embd, labels)):
    hidden = [item[1] for item in embed]
    if np.isnan(l):
        print("NaN in label %s encountered, continue." %(i, ))
        continue
    x[count] = np.array(hidden)
    y[count] = l
    count += 1

# %%
# s = np.diag(1/lsi.projection.s)
x = x[:count]
# x = np.matmul(x ,s)
y = y[:count]
# %%
np.savez("./data/tfidf_svd_%s.npz" %(embed_size, ), x = x, y = y)