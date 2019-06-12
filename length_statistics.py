from gensim.models import keyedvectors, phrases
import  logging
import fopen
import  sys

log = logging.getLogger(name= __name__)
log.setLevel(logging.DEBUG)
console = logging.StreamHandler(stream= sys.stdout)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
log.addHandler(console)

def LengthStat():
    """

    :return: max_fname, max_comment_list and max_len
    """
    # word_vector = keyedvectors.KeyedVectors.load('word2vec_models/model_wv')
    bigram = phrases.Phraser.load('word2vec_models/bigram_tst')
    # load test:
    # log.debug(word_vector['good'])
    # log.info('WordVector loaded succeed!')
    max_len = 0
    for line  in fopen.Sentences('aclImdb/train/pos', False):
        comment = bigram[line]
        print(len(comment))
        max_len = len(comment) if len(comment) > max_len else max_len
        max_comment = comment
    print('Max_length:', max_len)
    return max_comment, max_len





if __name__ == '__main__':
    # fname, comment, leng = LengthStat()
    # print(fname, comment, leng,sep= '\n')
    bigram = phrases.Phraser.load('word2vec_models/bigram_tst')
    with open('aclImdb/train/pos/1175_9.txt','r') as f:
        for line in f:
            comment = [word for word in line.lower().split() if word.isalpha()]
            comment = bigram[comment]
            print(comment)
            print(len(comment))
#             到头来有个评论有 2080 个字， 截断是不可避免，这个例子充分说明了在设计算法前必须对数据集的概况有所了解。
