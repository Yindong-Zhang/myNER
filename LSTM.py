from keras.layers import LSTM, Masking, Dense, Input, Embedding, BatchNormalization, GRU, Bidirectional, \
    GRUCell, StackedRNNCells, RNN, LSTMCell, Concatenate, Lambda
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from gensim.models import keyedvectors
from gensim.models import phrases
from gensim.corpora import Dictionary
import pandas as pd
import numpy as np
import os.path
from fopen import Sentences
from time_wrapper import func_timer
from config import log, EMBEDDING_LENGTH, SAVED_WORD2VEC_PATH, SAVED_BIGRAM_PATH
import gc
from itertools import product
from collections import OrderedDict
from keras.backend.tensorflow_backend import clear_session

def gridsearch(para_gird):
    '''
    This function take a parameter grid which is a list of parameter dictionary {param_key: param_value_list}
    :param para_gird:
    :return: an parameter grid search iterator
    '''
    for hyperpara in para_gird:
        keys = hyperpara.keys()
        values_list = hyperpara.values()
        # log.debug('Keys: %s \n Values: %s' %(keys, values_list))
        for values in product(*values_list):
            yield OrderedDict(zip(keys, values))

def test_data_generator(maxlen, totalsize):
    comments = np.random.random(size=(totalsize ,maxlen))
    sentiment = np.random.random(size=(totalsize, 2))
    return comments, sentiment


def sentencevectors(filepath, maxlen, totalsize):
    """
    该函数载入sentences 数据,通过 查找表 转化为已训练好的 word2vec 向量.
    :param filepath:
    :param maxlen:
    :param totalsize:
    :return: 返回一个 shape 为 (totalsize, maxlen, wordvec_size) 的 NumPy ndarray.
    """
    word_vector = keyedvectors.load('word2vec_models/model_wv')
    bigram = phrases.Phraser.load('word2vec_models/bigram_tst')
    sentences = Sentences(filepath, False)
    word_vec = np.zeros(shape=(totalsize, maxlen, 100), dtype= np.float32)
    sentiment = np.zeros(shape= (totalsize, 2), dtype= np.float32)
    for count, (fname, line) in enumerate(sentences):
        log.debug(line)

        grade = int(fname.split(sep= '_')[1].split(sep='.')[0])
        if grade < 5:
            sentiment[count, 0] = 1
        else:
            sentiment[count, 1] = 1
        # 新式的 .format() 不受支持， 旧式的 % 反而受支持。
        log.debug('grade: %+4s. sentiment: %+4s.' % (grade, sentiment[count, :]))


        j = 0
        for word in bigram[line]:
            try:
                log.debug('i: {:>4}. j: {:>4}. '.format(count, j))
                word_vec[count, j, :] = word_vector[word]
                j += 1
            except KeyError:
                log.debug('KeyError occured')
                # input('Press any key to continue.')
                continue
            finally:
                if j == maxlen:
                    # log.debug(i)
                    break
        if count == totalsize - 1:
            log.debug('training data shape: %+8s. \t training label shape: %+6s' % (word_vec.shape, sentiment.shape))
            return word_vec, sentiment



def load_comments(filepath, totalsize, maxlen, dictlen):
    wordvec = keyedvectors.KeyedVectors.load(SAVED_WORD2VEC_PATH)
    vocabulary = wordvec.vocab
    log.info('Length of vocabulary: %s' % len(vocabulary))
    index2word = wordvec.index2word
    log.info('The first %d most frequent words selected, word with frequency under %d discarded.' %(dictlen, vocabulary[index2word[dictlen]].count))
    assert dictlen == vocabulary[index2word[dictlen]].index

    bigramer = phrases.Phraser.load(SAVED_BIGRAM_PATH)

    sentences = Sentences(filename= filepath, loop= False)
    sentiments = np.zeros(shape= (totalsize, 1), dtype= np.float32)
    comments = np.zeros(shape= (totalsize, maxlen), dtype= np.float32)
    labels = pd.read_csv('./data/train_input.csv', usecols = ['label', ], squeeze = True)
    count = 0
    full = 0
    empty = 0
    for label, sentence in zip(labels, sentences):
        if len(sentence) == 0:
            print("Zero length sentences encountered: %s" %(count, ))
            continue

        if np.isnan(label):
            print("Nan label encountered: %s" %(count, ))
            continue
        sentiments[count, 0] = label
        sentence= bigramer[sentence]


        log.debug('Length of sentence: %+3s' % len(sentence))
        j = 0
        for word in sentence:
            word_info = vocabulary.get(word)
            if word_info:
                word_id = word_info.index
                if word_id < dictlen:
                    comments[count, j] = word_id + 1
                    log.debug('The %+5s-th wordvector[%+2s] in one-hot representation: %+5s' % (count, j, word_id))
                    j += 1
            else:
                continue
            if j == maxlen:
                full += 1
                break

        if j == 0:
            print("Sentence %s is empty after processing." %(count, ))
            empty += 1
        else:
            # if 1403 < count < 1409:
            #     print(count, sentence, comments[count])
            count += 1
        if count == totalsize - 1:
            break
    log.info('%+5s out of %+5s in total are full while %s are empty.' % (full, totalsize, empty))
    # print(comments[1401:1408])
    return comments[:count], sentiments[:count]

def build_embedding_weight(dictlen, embedding_dim = EMBEDDING_LENGTH):
    """
    该函数从word2vec 模型中加载 训练好的 embedded vectors.
    :param dictlen:
    :param embedding_dim:
    :return:
    """
    if embedding_dim != EMBEDDING_LENGTH:
        raise TypeError('word embedding length in model conflict with that in word2vec model')

    log.debug(SAVED_WORD2VEC_PATH)
    wordvec = keyedvectors.KeyedVectors.load(SAVED_WORD2VEC_PATH)
    vocabulary = wordvec.index2word
    # 注意 vocabulary 是以 单词频率 降序排列的
    freq_dict = wordvec.vocab
    weight = np.zeros(shape = (dictlen + 1, embedding_dim),)
    for i in range(dictlen):
        weight[i + 1,:] = wordvec[vocabulary[i]]
        if( i % 1000 == 0):
            print('word %+12s frequency: %+6s' %(vocabulary[i], freq_dict[vocabulary[i]]))
    return [weight,]






class GRU2Stacked():
    def __init__(self, rnn_units_1, rnn_units_2, return_sequences = True, return_state = False, dropout = 0., recurrent_dropout = 0., unroll = True, name = None):
        self.rnn_units_1 = rnn_units_1
        self.rnn_units_2 = rnn_units_2
        self.return_sequence = return_sequences
        self.return_state = return_state
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.unroll = unroll
        self.name = name

    def __call__(self, input):
        rnn_cell_1 = GRUCell(units=self.rnn_units_1, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name=self.name + '_rnn_cell_1' if self.name else None)
        rnn_cell_2 = GRUCell(units=self.rnn_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name=self.name + '_rnn_cell_2' if self.name else None)
        # gru_cell_3 = GRUCell(units= rnn_units_3, dropout= rnn_dropout, recurrent_dropout= rnn_recurrent_dropout, reset_after= False)
        rnn_stack_cell = StackedRNNCells(cells=[rnn_cell_1, rnn_cell_2], name=self.name + '_stacked_rnn_cell' if self.name else None)
        rnn = RNN(cell=rnn_stack_cell, return_state=self.return_state, return_sequences=self.return_sequence, unroll=self.unroll, name=self.name)(input)
        return rnn

class Comprehension_GRU():
    """
    将这个递归神经单元命名为 comprehension~~~
    """
    def __init__(self,
                 stack_2_units_1,
                 stack_2_units_2,
                 stack_1_units,
                 name= None,
                 dropout= 0.,
                 recurrent_dropout= 0.,
                 return_sequence= True,
                 unroll= True ):
        """

        :param stack_2_units_1:
        :param stack_2_units_2:
        :param stack_1_units:
        :param name:
        :param dropout:
        :param recurrent_dropout:
        :param return_sequence:

        """
        self.stack_2_units_1 = stack_2_units_1
        self.stack_2_units_2 = stack_2_units_2
        self.stack_1_units = stack_1_units
        self.return_sequence = return_sequence
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.unroll = unroll
        self.name = name

    def __call__(self, input):
        """
        默认不返回 state, 为简单起见.
        :param input:
        :return:
        """
        recurrent_2 = GRU2Stacked(rnn_units_1= self.stack_2_units_1,
                                  rnn_units_2= self.stack_2_units_2,
                                  return_sequences= self.return_sequence,
                                  # return_state= self.return_state,
                                  dropout= self.dropout,
                                  recurrent_dropout= self.recurrent_dropout,
                                  unroll= self.unroll,
                                  name= self.name + '_stacked_2' if self.name else None)(input)
        recurrent_1 = GRU(units= self.stack_1_units,
                          return_sequences= self.return_sequence,
                          # return_state= self.return_state,
                          dropout= self.dropout,
                          recurrent_dropout= self.recurrent_dropout,
                          unroll= self.unroll,
                          name= self.name + '_stacked_1' if self.name else None)(input)
        if self.return_sequence:
            output = Concatenate(axis= 2, name= self.name + '_Concatenate' if self.name else None)([recurrent_2, recurrent_1, input])
        else:
            output = Concatenate(axis= 1, name= self.name + '_Concatenate' if self.name else None)([recurrent_2, recurrent_1])
        return output
    
    
def build_model(maxlen= 128,
                embedding_dim= 64,
                dict_len= 4096,
                stack_2_units_1 = 32,
                stack_2_units_2 = 8,
                stack_1_units = 16,
                # rnn_units_1= 16,
                # rnn_units_2= 8,
                # rnn_units_3 = 16,
                # rnn_units_4 = 8,
                rnn_dropout= 0.25,
                recurrent_dropout= 0.25,
                embedding_l1= 0.0005,
                dense_units_1 = 10,
                lr = 1E-6,
                ):
    """
    这是一个DSN, 新设计一个越层连接,希望能提高 正确率
    :param maxlen:
    :param embedding_dim:
    :param dict_len:
    :param rnn_units_1:
    :param rnn_units_2:
    :param rnn_dropout:
    :param recurrent_dropout:
    :param embedding_l1:
    :param dense_units_1:
    :return:
    """
    # legacy

    input = Input(shape= (maxlen,),name= 'Input', dtype= np.int32)
    # model_input = Lambda(lambda x: K.print_tensor(x, message= "model_input: "), name= "print")(input)
    embedded = Embedding(input_dim= dict_len + 1, output_dim= embedding_dim, embeddings_regularizer= regularizers.l1(l = embedding_l1), name='Embedding', mask_zero= True,
                         weights= build_embedding_weight(dict_len, embedding_dim)
                         )(input)
    # embedded = Lambda(lambda x: K.print_tensor(x, message= "after_embedding: "), name= "print_1")(embedded)


    # concated = Concatenate(axis= 2)([embedded, rnn_1])
    # rnn_1 = Comprehension_GRU(stack_1_units= stack_1_units, stack_2_units_1= stack_2_units_1, stack_2_units_2 = stack_2_units_2, name= 'Comprehension_1', dropout= rnn_dropout, recurrent_dropout= recurrent_dropout, return_sequence= False)(embedded)
    rnn_1 = GRU(units= stack_1_units,
                  return_sequences= False,
                  dropout= rnn_dropout,
                  recurrent_dropout= recurrent_dropout,
                  unroll= True,
                  name= 'rnn_1')(embedded)

    # bn = BatchNormalization(axis= 1, center= False, scale= True)(lstm)
    bn_1 = BatchNormalization(axis= 1, center= False, scale= False, name= 'BatchNormaliazation_After_RNN_1')(rnn_1)
    dense_1 = Dense(units= dense_units_1, activation= 'tanh', name= 'Dense_1')(bn_1)
    bn_2 = BatchNormalization(axis= 1, center= False, scale= False, name= 'BatchNormalization_After_RNN_2')(dense_1)
    sent = Dense(units= 1, activation= 'sigmoid', name='Dense_2')(bn_2)

    model = Model(inputs= input, outputs= sent)
    adam = Adam(lr, clipnorm = 10, clipvalue = 5)
    model.compile(optimizer= adam, loss= 'binary_crossentropy', metrics= ['binary_accuracy'])
    model.summary()
    # input('Enter any key to continue...')
    return model



def make_config_str(kwargs):
    string = '-'.join(['%s~%s' %(key, kwargs[key]) for key in kwargs])
    return string


@func_timer
def main():

    # build a keras model;
    # 所有的参数都在这里～～
    # 以下为载入数据时的超参数
    totalsize = 6328
    maxlen = 32
    embedding_dim = EMBEDDING_LENGTH
    dictlen = 3600

    # 以下为构建模型的超参数
    hyperparas = [
        # 注意, 关键字参数在参数传递时似乎会先被转化为字典, 然后通过 **kwargs 传入.
        # 也就是 关键字参数 和 字典参数 等价
        OrderedDict(
            [
                ('maxlen', [maxlen, ]),
                # embedding_dim 应该与 word2vec 模型中的 维数 相同
                ('embedding_dim', [embedding_dim, ]),
                ('dict_len', [dictlen, ]),
                # ('rnn_units_1', [32, ]),
                # ('rnn_units_2', [24, ]),
                ('stack_2_units_1', [12,]),
                ('stack_2_units_2', [4, ]),
                ('stack_1_units', [6,]),
                ('dense_units_1', [10, ]),
                # ('rnn_units_3', [8,]),
                ('rnn_dropout', [0.4, ]),
                ('recurrent_dropout', [0.25, ]),
                ('embedding_l1', [1E-7,  ]),
                ("lr", [1E-2, ])

            ]
        ),

    ]

    # 以下为模型训练时所需超参数
    val_split=0.2
    reduceLR_factor=0.25

    # 以下为其他参数
    batchsize = 32
    epochs = 256

    # 加载数据
    log.info('Starting loading data...')
    comments, sentiment = load_comments(
        filepath= "./data/train_input.csv", totalsize= totalsize, maxlen = maxlen, dictlen= dictlen)
    log.info('Loading data conclude.')

    # 测试数据：
    # comments, sentiment = test_data_generator(maxlen= maxlen, totalsize= 2000)
    # comments_val, sentiment_val = test_data_generator(maxlen= maxlen,totalsize= 1000)

    for kwargs in gridsearch(hyperparas):
        log.info('keyword args: %s' % kwargs)
        config_str = make_config_str(kwargs)
        config_str += '-Comprehension_DSN'

        # 建立模型
        model = build_model(**kwargs)

        # 训练模型
        log.info('Start fitting:')
        model.fit(x= comments,
                  y= sentiment,
                  batch_size= batchsize,
                  epochs= epochs,
                  callbacks= [TensorBoard(log_dir= './logs_2/' + config_str, batch_size= batchsize, write_graph= False),
                              EarlyStopping(monitor= 'val_binary_accuracy', patience= 5, mode= 'max'),
                              ReduceLROnPlateau(monitor= 'val_loss', factor= reduceLR_factor, patience= 3, mode= 'min', cooldown= 0)
                              ],
                  # validation_data= (comments_val, sentiment_val),
                  validation_split= val_split,
                  shuffle= True,
                  )
        log.info('fitting conclude.')

        # 保存模型
        model.save( 'saved_models/' + config_str )
        clear_session()





if __name__ == '__main__':
    # 仅供测试用,不应该包含具体的业务逻辑.
    # data, sentiment = SentenceVectors(dirpaths= ['aclImdb/train/pos','aclImdb/train/neg'],
    #                                   maxlen= 60,
    #                                   totalsize= 25000)
    # print('training data shape: %+8s. \t training label shape: %+6s' %(data.shape, sentiment.shape))

    # comments = Comments(dirpaths= ['aclImdb/train/pos','aclImdb/train/neg'], maxlen= 60, totalsize= 30)
    # test = OrderedDict(
    #     a = [1],
    #     b = [2, 5],
    #     c = [3],
    #     d = [4],
    #         )
    # for kwargs in gridsearch(test_data):
    #     log.debug(kwargs)
    #     log.debug(make_config_str(kwargs))

    # Orderdict 以字典参数/关键字参数 传入是无效的, 字典本身就是无序的.
    # test = [OrderedDict(
    #     [
    #         ('a', [1, ]),
    #         ('b', [2, ]),
    #         ('c', [3, ]),
    #         ('d', [4, ]),
    #     ]
    # )
    #     ]
    # print(test[0])
    # print(test[0].keys())
    # print(test[0].values())
    # for kwargs in gridsearch(test):
    #     log.debug(kwargs)
    #     log.debug(make_config_str(kwargs))


    # totalsize = 50000
    # maxlen = 128
    # embedding_dim = 64
    # dictlen = 32768
    #
    # hyperparas = [
        # 注意, 关键字参数在参数传递时似乎会先被转化为字典, 然后通过 **kwargs 传入.
        # 也就是 关键字参数 和 字典参数 等价
        # OrderedDict(
        #     [
        #         ('maxlen', [maxlen, ]),
        #         embedding_dim 应该与 word2vec 模型中的 维数 相同
                # ('embedding_dim', [embedding_dim, ]),
                # ('dict_len', [dictlen, ]),
                # ('rnn_units_1', [32, ]),
                # ('rnn_units_2', [24, ]),
                # ('stack_2_units_1', [32,]),
                # ('stack_2_units_2', [16,]),
                # ('stack_1_units', [24,]),
                # ('dense_units_1', [10, ]),
                # ('rnn_units_3', [8,]),
                # ('rnn_dropout', [0.4, ]),
                # ('rnn_recurrent_dropout', [0.25, ]),
                # ('embedding_regularizer_l1', [0., ]),

            # ]
        # ),
    # ]
    # for hyperpara in gridsearch(hyperparas):
    #     build_model(**hyperpara)


    # weight = build_embedding_weight(100, EMBEDDING_LENGTH)
    # comments, sentiments = load_comments('aclImdb/train/pos', maxlen= 256, totalsize= 10, dictlen= DICTLENGTH)
    # log.info(comments.shape)
    main()
    # build_model()