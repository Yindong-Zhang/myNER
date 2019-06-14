import logging, sys
logging.basicConfig(format='%(asctime)s %(funcName)-20s: %(levelname)+6s: %(message)s', level=logging.INFO,)

# 0 for debug, 1 for run
develop_stage = 1

if develop_stage == 0:
    loglevel = logging.DEBUG
elif develop_stage == 1:
    loglevel = logging.INFO

log = logging.getLogger(name='global')
log.setLevel(loglevel)
console = logging.StreamHandler(stream= sys.stdout)
formatter = logging.Formatter('%(asctime)s %(funcName)-20s: %(levelname)+6s: %(message)s')
console.setLevel(loglevel)
console.setFormatter(formatter)
log.addHandler(console)
log.propagate = False

NUMSAMPLES = 6329
DICTLENGTH = 8192
EMBEDDING_LENGTH = 16
SVDSIZE= 360
SAVED_DICT_PATH = 'word2vec_models/test.dictionary'
SAVED_BIGRAM_PATH = 'word2vec_models/bigrammer'
SAVED_WORD2VEC_PATH = 'word2vec_models/words_embedded_in_%d_dimension.word2vec' % EMBEDDING_LENGTH


if __name__ == '__main__':
    log.info('test')
    print(log.handlers)
    print(log.parent)