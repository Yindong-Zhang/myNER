from itertools import product
from collections import OrderedDict

def grid_search(para_gird):
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