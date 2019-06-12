from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import pandas as pd
from time_wrapper import func_timer
import os
import random
from itertools import cycle
from config import log
import re
# from timeit import Timer



@func_timer
def fopen(filepath):
    '''
    read file aclImdb/train/labeledBow.feat to a scipy csr object
    '''
    count = 0
    x_spa = dok_matrix((25000, 89527),dtype= int)
    labels = np.ndarray(shape=(25000,),dtype= int)
    print('fopen start:')
    with open(filepath,'r') as f:
        for i, line in enumerate(f):
            # print(line)
            label, features = line.split(sep= ' ', maxsplit= 1)
            labels[i] = 0 if int(label) < 5 else 1
            for feature in features.split(sep = ' '):
                ind, value = feature.split( sep = ':')
                # print(ind, ':', value)
                x_spa[i, int(ind)] = int(value)
            # print(label, dok[i,:])
            # count += 1
            # if count == 3:
            #     break
        # x_spa = x_spa.tocsr()
        print('fopen succeed! ')
        return x_spa , labels

class Sentences():
    def __init__(self, filename, loop):
        """

        :param dirnames:
        :param loop: bool 类型, 决定是否循环利用迭代器
        """
        self.filename = filename
        self.loop = loop
        self.sentences = pd.read_csv(filename, usecols= ['comment', ], squeeze= True)

    # 这不是一个迭代器，但是一个可迭代对象。
    # 通过 self.loop 标志 确定是否循环迭代
    # 调用 self.__iter__ 得到一个生成器, 提供了一种复用生成器的方法.!!!
    def __iter__(self):
        if self.loop:
            self.sentences = cycle(self.sentences)
        for line in self.sentences:
            wordlist = re.findall("\w+'\w+|\w+|[,.:?]", line)
            yield wordlist

def corpora(filename, loop_or_not):
    """
    这是一个生成器函数, 返回一个生成器, 实现了 迭代器 协议.
    :param dirpaths:
    :param loop_or_not:
    :return:
    """
    sentences = Sentences(filename= filename, loop = loop_or_not)
    for sentence in sentences:
        yield sentence


if __name__ == '__main__':
    # x, y = fopen()
    # xi, yi = Iter_open('aclImdb/test/labeledBow.feat').__next__()
    # print(fopen.__doc__)
    # print(Iter_open.__doc__)
    # # print(x.shape, y.shape)
    # print(xi, yi)

    # test_comment = '''I haven't read a biography of Lincoln, so maybe this was an accurate portrayal......<br /><br />And maybe it's because I'm used to the equally alienating and unrealistic worshiping portrayals that unnaturally deify Lincoln as brilliant, honorable, and the savior of our country......<br /><br />But why would they make a movie representing Lincoln as a buffoon? While Henry Fonda made an excellent Lincoln, his portrayal of him as an "aw shucks, I'm just a simple guy" seemed a little insulting.<br /><br />[Granted, that was Bushie Jr.'s whole campaign, to make us think he was "just a regular guy" so we wouldn't care that he's a rich & privileged moron -- but that's a whole other story.]<br /><br />Not only did the film show Lincoln as sort of a simple (almost simple-minded) kind of guy , the film states that Lincoln just sort of got into law by accident, and that he wasn't even that interested in the law - only with the falsely simplistic idea of the law being about rights & wrongs. In the film he's not a very good defense attorney (he lounges around with his feet on the table and makes fun of the witnesses), and the outcome is mostly determined by chance/luck.<br /><br />Furthermore, partly because this was financed by Republicans (in reaction to some play sponsored by Democrats that had come out) and partly because it was just the sentiment of the times, the film is unfortunately religious, racist and conservative.<br /><br />Don't waste your time on this film.'''
    # test_comment = re.sub('<br\s*/>', '', test_comment)
    # log.debug('comment after substitute: %s.' %test_comment)
    # wordlist = [word for word in re.findall("\w+'\w+|\w+|[,.:?]", test_comment)]
    # log.debug('comment before split: %s' %test_comment)
    # log.debug('comment after split: %s' % wordlist)
    # log.debug('comment after split: %s' % ' '.join(wordlist))

    sentences = Sentences('./data/train_input.csv', loop= True)
    for count, sentence in enumerate(sentences):
        print(len(sentence))
        print(' '.join(sentence))
        if count == 20:
            break
