from fopen import Sentences, corpora
from gensim.models import phrases, word2vec, KeyedVectors
from gensim.corpora import Dictionary
from config import SAVED_BIGRAM_PATH, SAVED_WORD2VEC_PATH, SAVED_DICT_PATH, DICTLENGTH, EMBEDDING_LENGTH
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def make_bigram(dirpaths):
    sentences = corpora(dirpaths, loop_or_not= False)
    print('Start phrasing:')
    phrase = phrases.Phrases(sentences, max_vocab_size= DICTLENGTH,  min_count = 1, threshold= 5,  common_terms={'of', 'and', 'the', 'with'})
    bigram = phrases.Phraser(phrase)
    bigram.save(SAVED_BIGRAM_PATH)
    print('bigram phraser saved conclude.')


class Bigramer():
    def __init__(self, filepath, loop):
        """
        该类是一个可迭代对象, 包装一个 corpora 迭代器,对其应用 bigram.

        :param filepath:
        :param loop:
        """
        self.filepath = filepath
        self.loop_or_not = loop
        self.bigramer = phrases.Phraser.load(SAVED_BIGRAM_PATH)

    def __iter__(self):
        sentences = corpora(self.filepath, loop_or_not= self.loop_or_not)
        print('reload bigram...')
        for sentence in sentences:
            yield self.bigramer[sentence]

def make_wordvec():
    """
    本函数进行 word2vec 训练, 并将对应的模型保存到 SAVED_WORD2VEC_PATH
    :return: None
    """
    filepath = './data/train_input.csv'
    bigram_comments = Bigramer(filepath, loop= False)

    vec_size = EMBEDDING_LENGTH

    print('start training word2vec...')
    model = word2vec.Word2Vec(size= vec_size,
                              window= 5,
                              sg= 0,
                              cbow_mean= 1,
                              hs= 0,
                              negative= 8,
                              sample= 1e-3,
                              workers= 4,
                              min_count= 2,
                              sorted_vocab= 1,)
    model.build_vocab(sentences= bigram_comments)
    model.train(bigram_comments, total_examples= model.corpus_count, epochs= 128, compute_loss= True)
    print('finish training.')
    if os.path.exists(SAVED_WORD2VEC_PATH):
        os.remove(SAVED_WORD2VEC_PATH)
    model.wv.save(SAVED_WORD2VEC_PATH)
    word_vectors = model.wv
    # print(type(word_vectors))
    word_vectors.save('word2vec_models/model_wv')
    print('WordVector saved.')

def frequency_histogram(wordvec):
    vocabulary = wordvec.index2word
    freq_dict = wordvec.vocab
    freq_list = [freq_dict[word].count for word in vocabulary]
    figure, ax = plt.subplots()
    ax.hist(freq_list, bins= 100, log= True, range= [0, 200000])
    ax.set(xlabel = 'frequency', ylabel = 'count', title = 'word frequency histogram')
    ax.grid()
    plt.show()



def comments2Onehotvector(dictlen):
    dirpaths =  ['aclImdb/train/pos','aclImdb/train/neg']
    corp = corpora(dirpaths =dirpaths, loop_or_not= False)
    dict = Dictionary(documents=corp)
    dict.filter_extremes(no_below=5, no_above=0.5, keep_n= dictlen)
    actual_dictlen = len(dict)
    if actual_dictlen < dictlen:
        print('%-5s lenght of dictionary wanted, while actual length of dictionary returned: %+5s' % (dictlen, actual_dictlen))
        input('Enter any key to continue.')
    else:
        print('Length of dictionary: %s' % actual_dictlen)
    sorted_dfs = sorted(dict.dfs.items(), key=lambda x: x[1],reverse= True)
    for rank, item in enumerate(sorted_dfs[:100]):
        print('The %-4s-th most frequent word: %+12s, which frequency is %+6s' % (rank, dict[item[0]], item[1]))
    for rank, item in enumerate(sorted_dfs[:-101:-1]):
        print('The %-4s-th least frequent words: %+12s, which frequency is %+6s' % (rank, dict[item[0]], item[1]))
    dict.save(SAVED_DICT_PATH)

def main():
    filepath = "./data/train_input.csv"
    make_bigram(filepath)
    bigram_comments = Bigramer(filepath, loop=False)
    for count, comment in enumerate(bigram_comments):
        print('After bigram: %s' % ' '.join(comment))
        if count == 20:
            break

    make_wordvec()
    wordvec = KeyedVectors.load(SAVED_WORD2VEC_PATH)
    dictionary = wordvec.vocab
    wordlist = wordvec.index2word
    for count, word in enumerate(reversed(wordlist)):
        print('%+16s count %-6d times.' % (word, dictionary[word].count))
        # 计数频率被隐藏在 keyedvector.vocab.count 里...
        if count == 100:
            break
    for count, word in enumerate(wordlist):
        print('%+16s count %-6d times.' % (word, dictionary[word].count))
        # 计数频率被隐藏在 keyedvector.vocab.count 里...
        if count == 100:
            break

    print('index2word: %s' % len(wordvec.index2word))
    print(wordvec.index2word is wordvec.index2entity)
    # frequency_histogram(wordvec)
    # log.info(kv.similar_by_word('beautiful'))

if __name__ == '__main__':
    main()