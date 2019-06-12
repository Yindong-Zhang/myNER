from fopen import  corpora
from gensim.corpora import Dictionary
from config import log
from config import SAVED_DICT_PATH
from config import DICTLENGTH




# 这是一个脚本


if __name__ == '__main__':
    dict_len = DICTLENGTH
    comments2Onehotvector(dict_len)
